import os
import sqlite3
import operator
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from typing_extensions import Annotated, TypedDict

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agente.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# --- SQLite connection ---

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tareas (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            descripcion TEXT    NOT NULL,
            completada  INTEGER DEFAULT 0,
            creada_en   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


# --- Tools ---

@tool
def guardar_tarea(descripcion: str) -> str:
    """Guarda una nueva tarea en la base de datos SQLite.

    Args:
        descripcion: Descripción de la tarea a guardar.
    """
    conn = _get_conn()
    cursor = conn.execute("INSERT INTO tareas (descripcion) VALUES (?)", (descripcion,))
    conn.commit()
    task_id = cursor.lastrowid
    conn.close()
    return f"Tarea guardada con ID {task_id}: {descripcion}"


@tool
def listar_tareas() -> str:
    """Lista todas las tareas de la base de datos SQLite."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, descripcion, completada FROM tareas ORDER BY creada_en DESC"
    ).fetchall()
    conn.close()
    if not rows:
        return "No hay tareas registradas."
    lines = ["Tareas:"]
    for id_, desc, done in rows:
        estado = "✓" if done else "○"
        lines.append(f"  [{estado}] ID {id_}: {desc}")
    return "\n".join(lines)


@tool
def completar_tarea(id: int) -> str:
    """Marca una tarea como completada en la base de datos SQLite.

    Args:
        id: ID de la tarea a completar.
    """
    conn = _get_conn()
    cursor = conn.execute("UPDATE tareas SET completada = 1 WHERE id = ?", (id,))
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected == 0:
        return f"No se encontró tarea con ID {id}."
    return f"Tarea {id} marcada como completada."


tools = [guardar_tarea, listar_tareas, completar_tarea]

llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(
    content=(
        "Eres un asistente útil con acceso a una base de datos de tareas. "
        "Puedes guardar nuevas tareas, listar las existentes y marcarlas como completadas."
    )
)


# --- Helpers ---

def _clean_messages(messages: list) -> list:
    """Elimina ToolMessages huérfanos y HumanMessages duplicados consecutivos."""
    clean = []
    last_human_content = None

    def has_pending_tool_call_id(tool_call_id: str) -> bool:
        for prev_msg in reversed(clean):
            if getattr(prev_msg, "tool_calls", None):
                for call in prev_msg.tool_calls:
                    cid = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                    if cid == tool_call_id:
                        return True
        return False

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if msg.content == last_human_content:
                continue
            clean.append(msg)
            last_human_content = msg.content
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id and has_pending_tool_call_id(msg.tool_call_id):
                clean.append(msg)
        else:
            clean.append(msg)
    return clean


def _has_pending_tool_calls(messages: list) -> bool:
    pending: set = set()
    for msg in messages:
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                cid = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                if cid:
                    pending.add(cid)
        if type(msg).__name__ == "ToolMessage":
            cid = getattr(msg, "tool_call_id", None)
            if cid:
                pending.discard(cid)
    return bool(pending)


# =============================================================================
# SUBGRAPH 1 — Resumen de conversación
# Claves compartidas con padre : messages (entrada), summary (entrada/salida)
# Estado PRIVADO               : clean_messages  ← no existe en el padre
#
# Nodo 1 filter_messages: limpia el historial → clean_messages (privado)
# Nodo 2 call_llm       : usa clean_messages + summary → actualiza summary
# =============================================================================

class SummarySubgraphState(TypedDict):
    messages: List[BaseMessage]       # compartida: recibida del padre
    summary: str                      # compartida: recibida del padre y devuelta
    clean_messages: List[BaseMessage] # PRIVADA: sólo vive dentro del subgrafo


class SummarySubgraphOutputState(TypedDict):
    summary: str             # escribe de vuelta al padre
    parallel_outputs: List[str]


def filter_messages_node(state: SummarySubgraphState):
    """Nodo 1: limpia mensajes quitando AIMessages con tool_calls y ToolMessages."""
    messages = state.get("messages") or []
    clean = _clean_messages(messages)
    clean = [
        msg for msg in clean
        if not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None))
        and not isinstance(msg, ToolMessage)
    ]
    return {"clean_messages": clean}


def call_llm_node(state: SummarySubgraphState):
    """Nodo 2: llama al LLM con los mensajes ya limpios para generar/ampliar resumen."""
    clean = state.get("clean_messages") or []
    summary = state.get("summary") or ""
    if not clean:
        return {"summary": summary, "parallel_outputs": []}

    prompt = (
        f"Resumen previo: {summary}\n\nAmplía el resumen con los nuevos mensajes:"
        if summary
        else "Crea un resumen de la conversación anterior:"
    )
    response = llm.invoke(clean + [HumanMessage(content=prompt)])
    return {
        "summary": response.content,
        "parallel_outputs": [f"[summary] {response.content[:100]}"],
    }


_summary_builder = StateGraph(SummarySubgraphState, output_schema=SummarySubgraphOutputState)
_summary_builder.add_node("filter_messages", filter_messages_node)
_summary_builder.add_node("call_llm", call_llm_node)
_summary_builder.add_edge(START, "filter_messages")
_summary_builder.add_edge("filter_messages", "call_llm")
_summary_builder.add_edge("call_llm", END)
summary_subgraph = _summary_builder.compile()


# =============================================================================
# SUBGRAPH 2 — Análisis de la conversación con insight LLM
# Clave compartida con padre : messages (entrada)
# Estado PRIVADO             : metrics  ← no existe en el padre
#
# Nodo 1 extract_metrics : cuenta tools, roles, mensajes → metrics (privado)
# Nodo 2 generate_insight: LLM genera insight de 1 frase sobre el patrón de uso
# =============================================================================

class AnalysisSubgraphState(TypedDict):
    messages: List[BaseMessage]  # compartida: recibida del padre
    metrics: dict                # PRIVADA: sólo vive dentro del subgrafo


class AnalysisSubgraphOutputState(TypedDict):
    parallel_outputs: List[str]


def extract_metrics_node(state: AnalysisSubgraphState):
    """Nodo 1: extrae métricas del historial de mensajes."""
    messages = state.get("messages") or []
    tool_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}

    for msg in messages:
        role = type(msg).__name__
        role_counts[role] = role_counts.get(role, 0) + 1
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "?")
                tool_counts[name] = tool_counts.get(name, 0) + 1

    return {
        "metrics": {
            "total_msgs": len(messages),
            "role_counts": role_counts,
            "tool_counts": tool_counts,
        }
    }


def generate_insight_node(state: AnalysisSubgraphState):
    """Nodo 2: LLM genera un insight de 1 frase sobre el patrón de uso."""
    metrics = state.get("metrics") or {}
    if not metrics:
        return {"parallel_outputs": ["[analysis] sin datos"]}

    prompt = (
        f"Dado este resumen de métricas de una conversación con un agente de tareas, "
        f"escribe UNA sola frase en español que describa el patrón de uso del usuario. "
        f"Sé directo y conciso.\n\nMétricas: {metrics}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"parallel_outputs": [f"[analysis] {response.content.strip()}"]}


_analysis_builder = StateGraph(AnalysisSubgraphState, output_schema=AnalysisSubgraphOutputState)
_analysis_builder.add_node("extract_metrics", extract_metrics_node)
_analysis_builder.add_node("generate_insight", generate_insight_node)
_analysis_builder.add_edge(START, "extract_metrics")
_analysis_builder.add_edge("extract_metrics", "generate_insight")
_analysis_builder.add_edge("generate_insight", END)
analysis_subgraph = _analysis_builder.compile()


# =============================================================================
# MAP-REDUCE — Evaluación paralela de tareas pendientes con Send API
#
# list_pending_tasks : gateway — lee DB → pending_tasks (estado intermedio)
# send_to_assess     : edge fn — devuelve [Send("assess_task", {...})] por tarea (MAP)
# assess_task        : nodo MAP — evalúa UNA tarea con LLM; se lanza N veces en paralelo
# consolidate_tasks  : nodo REDUCE — agrega todos los assess_task en task_report
# =============================================================================

class TaskAssessState(TypedDict):
    """Estado privado de cada invocación Send — no existe en el grafo padre."""
    task_id: int
    task_desc: str


def list_pending_tasks(state: "State"):
    """Lee tareas pendientes de la DB y las guarda en pending_tasks."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, descripcion FROM tareas WHERE completada = 0 ORDER BY id"
    ).fetchall()
    conn.close()
    return {"pending_tasks": [{"id": r[0], "desc": r[1]} for r in rows]}


def send_to_assess(state: "State"):
    """Edge function: un Send por tarea → paralelo dinámico (MAP phase)."""
    tasks = state.get("pending_tasks") or []
    if not tasks:
        return ["consolidate_tasks"]  # sin tareas: saltar directo al reduce
    return [Send("assess_task", {"task_id": t["id"], "task_desc": t["desc"]}) for t in tasks]


def assess_task(state: TaskAssessState):
    """Nodo MAP: evalúa urgencia/prioridad de UNA tarea (N copias corren en paralelo)."""
    response = llm.invoke([
        HumanMessage(
            content=(
                f"Evalúa en UNA frase la urgencia y prioridad de esta tarea: "
                f"'{state['task_desc']}'. Indica prioridad alta/media/baja y razón."
            )
        )
    ])
    return {
        "task_assessments": [
            f"[ID {state['task_id']}] {state['task_desc']}: {response.content.strip()}"
        ]
    }


def consolidate_tasks(state: "State"):
    """Nodo REDUCE: agrega todos los resultados del MAP en un informe de prioridad."""
    assessments = state.get("task_assessments") or []
    if not assessments:
        return {"task_report": "No hay tareas pendientes."}
    joined = "\n".join(assessments)
    response = llm.invoke([
        HumanMessage(
            content=(
                "Dado este análisis de tareas pendientes, escribe un informe de 3-4 líneas "
                "con las tareas ordenadas de mayor a menor prioridad y una recomendación de siguiente acción:\n\n"
                + joined
            )
        )
    ])
    return {"task_report": response.content.strip()}


# =============================================================================
# GRAFO PADRE
# Flujo:
#   START → agente
#   agente → tools (si hay tool_calls) → agente  [bucle]
#   agente → [summary_subgraph ‖ analysis_subgraph ‖ list_pending_tasks]  (fan-out)
#   list_pending_tasks → send_to_assess → assess_task×N → consolidate_tasks → END
# =============================================================================

class State(MessagesState):
    summary: str
    parallel_outputs: Annotated[List[str], operator.add]
    pending_tasks: List[dict]                              # gateway → send_to_assess
    task_assessments: Annotated[List[str], operator.add]  # acumulado de assess_task (MAP)
    task_report: str                                       # resultado final (REDUCE)


def agente(state: State):
    clean = _clean_messages(state["messages"])
    summary = state.get("summary") or ""
    context = [SystemMessage(content=f"Resumen anterior: {summary}")] if summary else [sys_msg]
    return {"messages": [llm_with_tools.invoke(context + clean)]}


def route_from_agente(state: State) -> list[str]:
    """Fan-out condicional: tools si hay llamadas pendientes, 3 ramas paralelas si no."""
    if _has_pending_tool_calls(state["messages"]):
        return ["tools"]
    return ["summary_subgraph", "analysis_subgraph", "list_pending_tasks"]


builder = StateGraph(State)
builder.add_node("agente", agente)
builder.add_node("tools", ToolNode(tools))
builder.add_node("summary_subgraph", summary_subgraph)
builder.add_node("analysis_subgraph", analysis_subgraph)
builder.add_node("list_pending_tasks", list_pending_tasks)
builder.add_node("assess_task", assess_task)
builder.add_node("consolidate_tasks", consolidate_tasks)

builder.add_edge(START, "agente")
builder.add_conditional_edges(
    "agente",
    route_from_agente,
    ["tools", "summary_subgraph", "analysis_subgraph", "list_pending_tasks"],
)
builder.add_edge("tools", "agente")
builder.add_edge("summary_subgraph", END)
builder.add_edge("analysis_subgraph", END)
builder.add_conditional_edges("list_pending_tasks", send_to_assess, ["assess_task", "consolidate_tasks"])
builder.add_edge("assess_task", "consolidate_tasks")
builder.add_edge("consolidate_tasks", END)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# Exporta imagen del grafo con estructura interna de subgrafos (xray=1)
_graph_png = os.path.join(os.path.dirname(__file__), "..", "graph.png")
with open(_graph_png, "wb") as _f:
    _f.write(graph.get_graph(xray=1).draw_mermaid_png())

# --- Bucle de terminal ---

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    config = {"configurable": {"thread_id": "44"}}
    print("Agente listo. Escribe 'salir' para terminar.\n")

    initial_input = {
        "messages": [
            HumanMessage(
                content=(
                    "Guarda las tareas 'Revisar presupuesto Q3' y 'Llamar al cliente Acme'. "
                    "Luego lista todo, marca como completada la tarea con ID 1, "
                    "y vuelve a listar para confirmar el cambio."
                )
            )
        ]
    }

    seen_msg_ids: set = set()
    for event in graph.stream(initial_input, config, stream_mode="values"):
        last_msg = event["messages"][-1]
        msg_key = getattr(last_msg, "id", None) or id(last_msg)
        if msg_key not in seen_msg_ids:
            seen_msg_ids.add(msg_key)
            last_msg.pretty_print()

    state = graph.get_state(config)
    parallel_outputs = state.values.get("parallel_outputs", [])
    if parallel_outputs:
        print("\nParallel outputs:")
        for output in parallel_outputs:
            print(f"- {output}")

    task_report = state.values.get("task_report", "")
    if task_report:
        print(f"\n{'─'*60}\nTask Report (map-reduce):\n{task_report}\n{'─'*60}")
