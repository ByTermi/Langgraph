import os
import sqlite3
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

# Ruta al archivo SQLite (no en memoria). Se crea en agente_sqlite/data/
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agente.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# --- Conexión a SQLite ---

def _get_conn() -> sqlite3.Connection:
    # Abre conexión y crea la tabla si no existe
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


# --- Tools: funciones que el LLM puede llamar ---

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

# LLM con las tools enlazadas para que pueda llamarlas
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

# Mensaje de sistema: define el comportamiento base del agente
sys_msg = SystemMessage(
    content=(
        "Eres un asistente útil con acceso a una base de datos de tareas. "
        "Puedes guardar nuevas tareas, listar las existentes y marcarlas como completadas."
    )
)


# Estado del grafo: hereda messages (lista de mensajes) y añade summary
class State(MessagesState):
    summary: str  # resumen acumulado de conversaciones anteriores


# --- Helpers ---

def _clean_messages(messages: list) -> list:
    # Elimina ToolMessages huérfanos (sin AIMessage con tool_calls previo).
    # Necesario porque SqliteSaver puede guardar estados intermedios con tool
    # messages que ya no tienen su tool_call correspondiente tras un summarize.
    clean = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # Solo conservar si el mensaje anterior es una llamada a tool
            if clean and isinstance(clean[-1], AIMessage) and clean[-1].tool_calls:
                clean.append(msg)
        else:
            clean.append(msg)
    return clean


# --- Nodos del grafo ---

def agente(state: State):
    summary = state.get("summary", "")
    # Si existe un resumen previo, se inyecta como contexto en lugar del sys_msg normal
    if summary:
        context = [SystemMessage(content=f"Resumen de la conversación anterior: {summary}")]
    else:
        context = [sys_msg]
    return {"messages": [llm_with_tools.invoke(context + _clean_messages(state["messages"]))]}


def should_continue(state: State) -> Literal["tools", "summarize_conversation", "__end__"]:
    last = state["messages"][-1]
    # Si el LLM quiere llamar una tool → ir al nodo tools
    if last.tool_calls:
        return "tools"
    # Si el historial supera 6 mensajes → resumir para no acumular tokens infinitamente
    if len(state["messages"]) > 6:
        return "summarize_conversation"
    # Si no → fin de la interacción
    return END


def summarize_conversation(state: State):
    summary = state.get("summary", "")
    # Si ya hay un resumen previo, se pide ampliarlo; si no, se crea uno nuevo
    if summary:
        prompt = (
            f"Este es el resumen de la conversación hasta ahora: {summary}\n\n"
            "Amplía el resumen teniendo en cuenta los nuevos mensajes anteriores:"
        )
    else:
        prompt = "Crea un resumen de la conversación anterior:"

    clean = _clean_messages(state["messages"])
    response = llm.invoke(clean + [HumanMessage(content=prompt)])

    # Borrar todos los mensajes excepto los 2 últimos para liberar memoria del estado
    delete = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete}


# --- Construcción del grafo ---

builder = StateGraph(State)
builder.add_node("agente", agente)
builder.add_node("tools", ToolNode(tools))             # ejecuta la tool que pidió el LLM
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "agente")                      # entrada → agente
builder.add_conditional_edges("agente", should_continue)  # agente → tools / summarize / END
builder.add_edge("tools", "agente")                    # tras ejecutar tool → volver al agente
builder.add_edge("summarize_conversation", END)        # tras resumir → fin

# SqliteSaver persiste el estado completo del grafo en disco (historial de mensajes,
# summary, etc.) usando el mismo archivo DB que las tools
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)

graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

# Guarda imagen del grafo en agente_sqlite/graph.png
graph_png = os.path.join(os.path.dirname(__file__), "..", "graph.png")
with open(graph_png, "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# --- Bucle de terminal ---

if __name__ == "__main__":
    # thread_id identifica la conversación; el mismo ID recupera el historial entre ejecuciones
    config = {"configurable": {"thread_id": "1"}}
    print("Agente listo. Escribe 'salir' para terminar.\n")

    while True:
        texto = input("Tú: ").strip()
        if not texto or texto.lower() == "salir":
            break

        result = graph.invoke({"messages": [HumanMessage(content=texto)]}, config)
        respuesta = result["messages"][-1].content
        print(f"Agente: {respuesta}\n")
