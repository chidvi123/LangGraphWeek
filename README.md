# LangGraph --- Complete Learning README

> **Scope:** This document is exclusively about LangGraph: graph
> structure, state, routing, agent workflows, and how we implemented and
> studied ReAct, Reflection, and a Reflexion-style agent. It also covers
> building RAG workflows with LangGraph, followed by Self-RAG and
> Corrective RAG (CRAG).

> Examples are learning patterns. Verify exact API signatures against
> the versions installed in your environment.

## Contents

1.  What LangGraph is
2.  Graph fundamentals
3.  State schemas
4.  Nodes
5.  Edges and graph execution
6.  `compile()` and `invoke()`
7.  State updates and reducers
8.  Message state and `MessagesState`
9.  Conditional routing
10. Loops and stopping conditions
11. ReAct agent
12. Tool execution and `ToolNode`
13. Reflection agent
14. Reflexion-style agent
15. Comparing ReAct, Reflection, and Reflexion
16. Checkpointing and thread persistence
17. Memory concepts
18. Streaming
19. Retries and error handling
20. Human-in-the-loop
21. RAG workflows in LangGraph
22. Self-RAG workflow
23. Corrective RAG (CRAG) workflow
24. GraphRAG and agentic workflows
25. Debugging checklist
26. KT and interview revision
27. Current status and next steps
28. Glossary

------------------------------------------------------------------------

# 1. What LangGraph Is

LangGraph is a framework for building stateful applications and
workflows whose execution is represented as a graph. The graph consists
of nodes that perform work and edges that determine where execution goes
next.

It is useful when an application needs: - A sequence of processing
steps. - Branches based on the current state. - Loops and repeated
attempts. - Tool-using agents. - Shared state across multiple steps. -
Checkpointing and resumable workflows. - Human review or approval. -
Explicit control over when the workflow stops.

A LangGraph workflow does not have to be an LLM agent. A graph can be
entirely deterministic, entirely model-driven, or a combination of both.

## 1.1 The central mental model

``` text
                 STATE
                   |
                   v
        +----------------------+
        |        NODE          |
        | Read state, do work  |
        | Return state updates |
        +----------------------+
                   |
                   v
                 EDGE
                   |
           +-------+-------+
           |               |
           v               v
       Next node        Another route
           |               |
           +-------+-------+
                   |
                   v
             Continue or END
```

The state carries data. Nodes process that data. Edges control
execution.

# 2. Graph Fundamentals

## 2.1 Core terms

-   **State:** The shared data structure used during graph execution.
-   **Node:** A function or runnable that reads state and returns
    updates.
-   **Edge:** A connection that determines the next node.
-   **`START`:** Special entry point for the graph.
-   **`END`:** Special terminal destination.
-   **`StateGraph`:** Builder used to define the graph and its state
    schema.
-   **`compile()`:** Prepares the graph as an executable runnable.
-   **`invoke()`:** Executes the compiled graph for an input.
-   **Conditional edge:** A route selected by a function based on state.
-   **Reducer:** Defines how an update is combined with the existing
    state value.
-   **Checkpointer:** Saves execution state so it can be resumed or
    recovered.

## 2.2 Basic graph

``` python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    name: str
    greeting: str

def greeting_node(state: State):
    return {"greeting": f"Hello, {state['name']}!"}

builder = StateGraph(State)
builder.add_node("greeting_node", greeting_node)
builder.add_edge(START, "greeting_node")
builder.add_edge("greeting_node", END)

app = builder.compile()

result = app.invoke({
    "name": "Learner",
    "greeting": "",
})

print(result)
```

### How the graph executes

1.  `State` describes the expected state fields.
2.  `greeting_node` is registered as a node.
3.  `START` points to `greeting_node`.
4.  The node reads `state["name"]`.
5.  It returns an update for `greeting`.
6.  The edge sends execution to `END`.
7.  `compile()` creates the runnable graph.
8.  `invoke()` runs the graph and returns the final state.

## 2.3 Fixed execution path

``` text
START -> node_a -> node_b -> node_c -> END
```

This is useful when every run should follow the same sequence.

## 2.4 Branching execution path

``` text
START -> classify
            |
       conditional route
          /       \
         v         v
      route_a    route_b
          \       /
             END
```

The branch is selected by a routing function or another supported
routing mechanism.

# 3. State Schemas

State defines the data available to nodes. It can hold strings, numbers,
lists, dictionaries, messages, flags, counters, and other supported
values.

## 3.1 `TypedDict`

``` python
from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    answer: str
    attempts: int
```

`TypedDict` documents the expected dictionary structure and helps static
type checking. It does not, by itself, perform complete runtime
validation.

## 3.2 Reading state

A node receives the current state as an argument:

``` python
def answer_node(state: State):
    question = state["question"]
    return {"answer": f"You asked: {question}"}
```

The node reads `question` and returns an update for `answer`.

## 3.3 Partial updates

Nodes usually return only the fields they need to update. If a node
returns:

``` python
return {"answer": "A graph contains nodes and edges."}
```

the update targets `answer`; it does not mean every other state field
should be erased. For ordinary fields, the new value replaces the
previous value for that field. Reducers can define different behavior.

## 3.4 State is not automatically persistent

State is available during the graph's execution. Persistence across
invocations requires an appropriate checkpointer and configuration.
Merely declaring a field called `memory` does not make it durable.

# 4. Nodes

A node performs one logical unit of work. A node may: - Read state. -
Call Python logic. - Call a model. - Execute a retrieval operation. -
Execute tools. - Evaluate a result. - Return one or more state updates.

## 4.1 Node signature

A common pattern is:

``` python
def my_node(state: State):
    # Read values from state.
    # Perform work.
    return {"some_field": "updated value"}
```

The returned keys must be compatible with the graph's state schema and
update behavior.

## 4.2 Registering nodes

``` python
builder.add_node("draft", draft_node)
builder.add_node("revise", revise_node)
```

The first argument is the graph node's name. The second is the callable
or runnable that performs the work.

## 4.3 Keep nodes focused

For a research-and-revision workflow, separate responsibilities might
be: - `draft`: produce the first answer. - `execute_tools`: execute
requested searches. - `revise`: use the answer and search results to
improve the answer. - `evaluate`: decide whether more work is needed.

Focused nodes make tracing, testing, and debugging easier.

# 5. Edges and Graph Execution

## 5.1 Fixed edges

``` python
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
```

A fixed edge always directs execution from the source node to the
destination node when that transition is reached.

## 5.2 Entry point

These two common approaches express the graph entry:

``` python
builder.add_edge(START, "draft")
```

or, in suitable graph patterns:

``` python
builder.set_entry_point("draft")
```

Use the style supported by the graph/API version and avoid defining
conflicting entry routes.

## 5.3 Graph execution is not a direct function call

If a node named `draft` is connected to `execute_tools`, the graph
scheduler moves to the tool node after `draft` finishes. The `draft`
function does not need to call the tool executor directly.

``` text
draft_node returns update
          |
          v
LangGraph follows edge
          |
          v
execute_tools node runs
```

This distinction is important when explaining a workflow during a KT
session.

# 6. `compile()` and `invoke()`

## `compile()`

``` python
graph = builder.compile()
```

Prepares the graph and returns an executable runnable. It does not run
the task itself.

## `invoke()`

``` python
result = graph.invoke(input_state)
```

Runs the compiled graph with the supplied input and returns the final
state/result.

## `stream()`

``` python
for event in graph.stream(input_state, stream_mode="updates"):
    print(event)
```

Yields intermediate execution events or updates according to the
selected stream mode.

### Remember

-   `compile()` = prepare the graph.
-   `invoke()` = execute and return the result.
-   `stream()` = execute and yield intermediate events.

# 7. State Updates and Reducers

## 7.1 Default update behavior

For a normal state field, a new update generally replaces the previous
value of that field. Fields not included in the update remain unchanged.

## 7.2 Reducer example

``` python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    notes: Annotated[list[str], operator.add]
```

Here `operator.add` specifies list concatenation when combining updates
to `notes`.

`Annotated` attaches reducer metadata to the field. It does not itself
perform the concatenation; LangGraph uses the reducer when processing
updates.

If the current value is `["first"]` and the node returns `["second"]`,
the resulting list is conceptually:

``` python
["first", "second"]
```

## 7.3 Why reducers matter

A workflow might collect: - Messages. - Search results. - Validation
errors. - Research notes. - Audit entries. - Intermediate outputs.

A reducer determines whether new data replaces or combines with previous
data.

## 7.4 Reducer caution

Do not choose concatenation when the field should hold only the latest
value. Do not assume every field is appended. The state annotation and
graph update semantics matter.

# 8. Message State and `MessagesState`

## 8.1 Message types

LangGraph workflows that interact with chat models commonly use: -
`HumanMessage`: User input. - `AIMessage`: Model output; can contain
tool calls. - `ToolMessage`: Result returned by a tool execution.

## 8.2 `MessagesState`

`MessagesState` is a predefined graph state schema with a `messages`
field and message-aware update behavior.

``` python
from langgraph.graph import StateGraph, MessagesState, START, END

def simple_node(state: MessagesState):
    return {"messages": []}

builder = StateGraph(MessagesState)
builder.add_node("simple", simple_node)
builder.add_edge(START, "simple")
builder.add_edge("simple", END)
graph = builder.compile()
```

This only illustrates registration and state shape; an empty message
update is not useful for an actual conversational task.

## 8.3 `add_messages`

A custom state can define the message field explicitly:

``` python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` supports appending messages and handling message IDs. It
is more message-aware than simple list concatenation.

## 8.4 Message history and prompts

A prompt may include the current history, but the prompt mechanism does
not independently persist that history. The graph state or another
persistence mechanism must supply it.

## 8.5 Message history is not the same as long-term memory

A graph can have a `messages` list without retaining it after the
process exits. Persistence and long-term memory require explicit design.

# 9. Conditional Routing

Conditional routing allows a graph to choose a destination based on
state.

## 9.1 Basic example

``` python
from typing import Literal
from langgraph.graph import END

def route(state) -> Literal["continue", END]:
    if state["is_complete"]:
        return END
    return "continue"
```

The function returns a destination label. That label must correspond to
a valid node or a destination in the route mapping.

## 9.2 `add_conditional_edges`

``` python
builder.add_conditional_edges(
    "evaluate",
    route,
    {
        "continue": "work",
        END: END,
    },
)
```

The exact mapping should match what the routing function returns.

## 9.3 Fixed edge versus conditional edge

  -----------------------------------------------------------------------
  Fixed edge                          Conditional edge
  ----------------------------------- -----------------------------------
  Predictable transition              Destination depends on a routing
                                      decision

  Good for mandatory steps            Good for branches, loops, and
                                      completion checks

  No router function needed           Uses a router or supported
                                      condition
  -----------------------------------------------------------------------

## 9.4 Built-in routing for tool calls

LangGraph's prebuilt `tools_condition` is commonly used in a ReAct-style
graph. It checks whether the latest AI message requests tool calls and
routes to the tool node or the end of the agent turn.

# 10. Loops and Stopping Conditions

Loops allow a graph to retry, revise, retrieve more evidence, or repeat
a tool cycle.

``` text
          +----------------------+
          |                      |
          v                      |
       work -> evaluate ---------+
                  |
                  +---- complete -> END
```

Every loop needs a well-defined stop condition.

## 10.1 Hard iteration limit

A hard limit prevents endless loops even if the evaluator never declares
success.

## 10.2 Quality-based completion

A separate evaluation signal can determine whether an answer is
complete, supported, or needs another attempt.

## 10.3 Prefer explicit counters

A dedicated integer field such as `iterations` is usually clearer than
counting unrelated messages to infer the number of iterations.

A robust design must: 1. Include the counter in the state schema. 2.
Initialize it. 3. Increment it at a clearly defined point. 4. Check it
in the router. 5. Stop when the configured maximum is reached. 6. Handle
tool failures and missing evaluation results.

## 10.4 Avoid off-by-one errors

With:

``` python
if count > MAX_ITERATIONS:
    return END
```

the condition becomes true only after the count exceeds the maximum. If
the intended behavior is to stop once the count reaches the maximum,
`>=` is the usual comparison. But the counter itself must measure the
correct thing.

# 11. ReAct Agent

## 11.1 What ReAct means

**ReAct = Reasoning + Acting.** The agent alternates between deciding
what to do, executing an action, observing its result, and deciding what
to do next.

A tool-using ReAct agent can: - Answer directly when no tool is
needed. - Request a tool call. - Observe tool results. - Call another
tool if necessary. - Produce a final answer.

## 11.2 ReAct loop

``` text
Human question
     |
     v
Agent/model node
     |
     +---- no tool call -----> END
     |
     +---- tool call --------> ToolNode
                                  |
                                  v
                             ToolMessage
                                  |
                                  v
                             Agent/model
                                  |
                             repeat or END
```

## 11.3 Main components

-   **Agent node:** Invokes the model with the current messages and the
    available tool definitions.
-   **Tool definitions:** Describe available operations and their input
    schemas.
-   **`ToolNode`:** Executes tool calls found in model output.
-   **`tools_condition`:** Routes according to whether tool calls are
    present.
-   **`MessagesState`:** Carries user, assistant, and tool messages
    through the loop.

## 11.4 Graph structure

``` python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "agent")

graph = builder.compile()
```

This is a structural example. `agent_node` and `tools` must be defined,
and the model must be configured to request the registered tools.

## 11.5 Tool calling versus tool execution

`bind_tools(tools)` normally exposes tool definitions to the model. It
does not by itself execute the Python functions.

``` text
AIMessage with tool call
          |
          v
ToolNode executes matching function
          |
          v
ToolMessage with result
          |
          v
Agent sees the result
```

## 11.6 Strengths and limits

Strengths: - Dynamic tool selection. - Access to APIs, search,
calculators, and databases. - Multi-step decisions based on
observations.

Limits: - Invalid tool names or arguments. - Bad tool choice. - Empty or
misleading results. - Excessive loops and cost. - Tool side effects. -
Untrusted content in tool output.

Use a hard iteration limit, validate inputs, handle errors, and protect
side-effecting tools.

# 12. Tool Execution and `ToolNode`

## 12.1 What `ToolNode` does

`ToolNode` is a prebuilt LangGraph node for executing tool calls made by
the model. It matches tool-call names against registered tools, executes
them, and returns tool-result messages.

``` python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

The node name used in the graph is the name registered through
`add_node`, for example:

``` python
builder.add_node("execute_tools", tool_node)
```

The Python variable name and graph node name are not required to be the
same.

## 12.2 Registered tool names and schemas

The tool-call name and arguments from the model must match the
registered tool definition. When a tool call fails, inspect: - Actual
`AIMessage.tool_calls`. - Tool name. - Tool input schema. - Supplied
arguments. - Function signature. - Tool result/error.

A Pydantic schema describing a final answer is not automatically the
correct schema for a search function.

## 12.3 Multiple tool calls

A model can request more than one tool call, depending on the model and
configuration. The number of `ToolMessage` objects therefore may not
equal the number of graph iterations. Never assume one tool result
equals one full agent loop.

# 13. Reflection Agent

## 13.1 Concept

A Reflection agent produces an initial output, critiques it, and uses
that critique to improve the output.

``` text
Generate -> Critique -> Revise
    ^                     |
    |---------------------|
         repeat if needed
```

Reflection is useful for improving completeness, clarity, reasoning
structure, and identifying omissions. It does not guarantee factual
correctness.

## 13.2 Code studied

``` python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    response = generation_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": [response]}

def reflection_node(state: MessageGraph):
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })
    return {"messages": [HumanMessage(content=response.content)]}

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.set_entry_point(GENERATE)
builder.add_node(REFLECT, reflection_node)
builder.add_conditional_edges(
    GENERATE,
    should_continue,
    path_map={END: END, REFLECT: REFLECT},
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
```

The `chains.py` prompt definitions must accept the input keys used here
and correctly handle the supplied message history.

## 13.3 Step-by-step execution

1.  `GENERATE` calls `generation_chain` with the full current history.
2.  The generated response is returned as a message update.
3.  `should_continue` checks the state.
4.  If the stop rule is not met, execution routes to `REFLECT`.
5.  `REFLECT` invokes `reflection_chain` with the current history.
6.  The critique is wrapped in a `HumanMessage` and added to state.
7.  The graph returns to `GENERATE`.
8.  The next generation can use the critique as feedback.
9.  The loop ends when the routing condition returns `END`.

## 13.4 Why wrap the critique as a `HumanMessage`?

It makes the critique appear as feedback in the conversation history for
the next generation. This is a design choice rather than a requirement
of every Reflection implementation. A dedicated state field or a
different message design can also work.

## 13.5 Stop-condition caveat

The code checks `len(state["messages"]) > 6`. This means it stops at
seven or more messages, not exactly six. Message count is a basic
demonstration; it is not a reliable measure of output quality or actual
iteration count.

# 14. Reflexion-Style Agent

## 14.1 Concept

The original Reflexion approach emphasizes learning from attempts
through verbal reflections or lessons that can guide later attempts. A
typical conceptual pattern includes: - An **actor** that attempts the
task. - An **evaluator** that assesses the attempt. - A
**self-reflection** step that produces actionable lessons. - **Memory**
that preserves useful lessons for future attempts.

The project we studied is best described as a **Reflexion-style
research-and-revision agent**. It drafts an answer, requests search
evidence, revises the answer, and may repeat the process. It does not
have a dedicated persistent episodic-memory component for lessons across
separate runs, so it is not a full implementation of the original
Reflexion architecture.

## 14.2 Main code studied

``` python
from dotenv import load_dotenv
from typing import Literal
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

load_dotenv()

MAX_ITERATIONS = 2

def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({
        "messages": state["messages"]
    })
    print(response.tool_calls)
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({
        "messages": state["messages"]
    })
    return {"messages": [response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine the route based on tool-result messages."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage)
        for item in state["messages"]
    )
    num_iterations = count_tool_visits

    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges(
    "revise",
    event_loop,
    ["execute_tools", END],
)

graph = builder.compile()
```

## 14.3 Graph flow

``` text
START
  |
  v
draft
  |
  v
execute_tools
  |
  v
revise
  |
  v
event_loop
  |                 \
  | continue         \ stop
  v                   v
execute_tools         END
  |
  v
revise
  |
  +---- event_loop ...
```

## 14.4 What each node does

### `draft_node`

Calls `first_responder` with the current messages and returns its
response. Printing `response.tool_calls` is useful for debugging whether
the model actually requested a tool.

### `execute_tools`

This is a graph node registered from the `execute_tools` object imported
from `tool_executor.py`. In the studied code, that object is a
`ToolNode` configured with tool definitions.

The graph's fixed edge from `draft` to `execute_tools` schedules that
node. `draft_node` does not directly invoke the tool executor.

### `revise_node`

Calls `revisor` with the current message history. It can use the draft
and tool results to improve the response. Whether it requests more tools
depends on how `revisor` is configured in `chains.py`.

### `event_loop`

Counts `ToolMessage` instances and decides whether to return to
`execute_tools` or finish.

## 14.5 Schemas studied

``` python
from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous.")

class AnswerQuestion(BaseModel):
    """Provide an initial answer and critique."""
    answer: str = Field(
        description="Only the answer, approximately 150 words. "
                    "Do not include reflection or search queries inside this field."
    )
    reflection: Reflection = Field(
        description="Critique the answer. Populate only missing and "
                    "superfluous; do not include search queries here."
    )
    search_queries: List[str] = Field(
        description="A top-level list containing 1-3 search queries "
                    "to improve the answer."
    )

class ReviseAnswer(AnswerQuestion):
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
```

### Schema interpretation

-   `answer`: The draft or revised answer.
-   `reflection.missing`: Missing content.
-   `reflection.superfluous`: Unnecessary content.
-   `search_queries`: Queries to use for research.
-   `references`: Sources intended to support the revised answer.

`ReviseAnswer` inherits the fields from `AnswerQuestion` and adds
`references`.

A schema defines the expected shape, but does not automatically make a
model use that shape. The chain must be configured to produce structured
output or tool calls accordingly.

## 14.6 Tool executor studied

``` python
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=1)

def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([
        {"query": query}
        for query in search_queries
    ])

execute_tools = ToolNode([
    StructuredTool.from_function(
        run_queries,
        name=AnswerQuestion.__name__,
    ),
    StructuredTool.from_function(
        run_queries,
        name=ReviseAnswer.__name__,
    ),
])
```

This records the implementation we studied. It needs an end-to-end test
against the installed package versions and the chain configuration.

## 14.7 Potential tool-schema mismatch

Both tools use the same underlying `run_queries` function but advertise
different names. `StructuredTool.from_function` generally infers an
input schema from the function signature. The function's primary
parameter is `search_queries`, whereas `AnswerQuestion` and
`ReviseAnswer` describe complete answer objects with fields such as
`answer`, `reflection`, and `references`.

Those are different schemas. If the model sends a full answer object to
a tool that expects only `search_queries`, execution can fail. Inspect
the actual registered tool schemas and `AIMessage.tool_calls` arguments.
Ensure the names and arguments the model emits match the registered
tools.

Two different tool names do not imply two different implementations when
both point to the same function.

## 14.8 Stop condition --- important

The code calculates:

``` python
count_tool_visits = sum(
    isinstance(item, ToolMessage)
    for item in state["messages"]
)
```

This counts tool-result messages, not true tool-node visits or full
revision cycles.

With:

``` python
MAX_ITERATIONS = 2

if num_iterations > MAX_ITERATIONS:
    return END
```

the condition stops only when the count exceeds 2: three or more
`ToolMessage` objects. Replacing `>` with `>=` would stop when the count
reaches 2, but it would still be a proxy for iterations. A batch or
multiple tool calls can produce a different number of result messages.

A better implementation uses a dedicated counter in the graph state and
increments it at a clearly defined point. It can also use a separate
evaluation signal for answer completion. Both the counter and evaluation
flag must be initialized and updated; adding fields to a schema alone
does not implement that logic.

## 14.9 Test input

``` python
from langchain_core.messages import HumanMessage

inputs = HumanMessage(content="""
Explain what LangGraph is and list 2 practical use cases.
Include reliable source URLs.
""")

result = graph.invoke({"messages": [inputs]})
```

The `MessagesState` input is a dictionary containing the `messages`
list.

## 14.10 What this agent demonstrates

-   Drafting an initial answer.
-   Requesting external search.
-   Executing tools with `ToolNode`.
-   Passing results through message state.
-   Revising a draft using gathered information.
-   Repeating under a stopping condition.

Before calling it production-ready, verify tool binding, schema
compatibility, source validity, iteration limits, final output format,
and error handling.

# 15. Comparing ReAct, Reflection, and Reflexion

  -------------------------------------------------------------------------
  Pattern             Main goal         Typical flow      What guides the
                                                          next step?
  ------------------- ----------------- ----------------- -----------------
  ReAct               Solve a task      Decide → act →    Tool observations
                      through actions   observe → decide  and current
                                                          context

  Reflection          Improve the       Generate →        Critique of
                      current output    critique → revise current output

  Reflexion           Learn from        Attempt →         Lessons/memory
                      attempts to       evaluate →        from prior
                      improve later     reflect → retry   attempts
                      attempts                            

  Reflexion-style     Improve an answer Draft → search →  Search evidence
  research/revision   with research     revise → repeat   and revision
  project                                                 feedback
  -------------------------------------------------------------------------

These patterns can be combined. A ReAct agent can use retrieval as a
tool, and Reflection can critique an answer produced with RAG.

# 16. Checkpointing and Thread Persistence

## 16.1 What checkpointing does

A checkpointer saves graph execution state at checkpoints. When
configured correctly, it can support resuming an interrupted workflow,
inspecting previous states, and maintaining thread state across
invocations.

## 16.2 Thread ID

A thread ID identifies a workflow thread when used with a checkpointer.

Conceptually:

``` python
config = {
    "configurable": {
        "thread_id": "conversation-001"
    }
}
```

The graph must be compiled/configured with an appropriate checkpointer
for thread persistence to work. Reusing the same thread ID with the same
checkpointer can continue the same thread's stored state.

A thread ID is not a security boundary or proof of user identity.

## 16.3 In-memory versus durable storage

-   **In-memory checkpointer:** Convenient for demos and tests; data is
    typically lost when the process exits.
-   **Durable checkpointer:** A supported persistent backend can
    preserve checkpoints across process restarts when configured
    properly.

Choose storage based on durability, deployment, security, and
operational requirements.

# 17. Memory Concepts

Memory is not one single feature. Distinguish these concepts:

-   **Current graph state:** Values available during an execution.
-   **Conversation history:** Messages stored in the graph state.
-   **Checkpointed thread state:** State saved by a configured
    checkpointer.
-   **Long-term memory:** Information deliberately stored and retrieved
    across separate conversations/tasks.
-   **Episodic memory:** Records of past attempts, outcomes, or lessons
    used to guide future attempts.

`MessagesState` does not automatically implement long-term memory. A
checkpointer can preserve thread state but does not, by itself, create a
general cross-thread knowledge store.

For a Reflexion system, persistent lessons should be deliberately
recorded, retrieved at the right time, and evaluated for relevance.
Otherwise the system may simply repeat the same mistakes or carry
irrelevant lessons forward.

# 18. Streaming

Streaming exposes intermediate execution events instead of waiting only
for the final result.

Common stream modes include the following, depending on the LangGraph
version and API: - `updates`: Node-level updates. - `values`: State
snapshots. - `messages`: Model message/token chunks and metadata. -
`custom`: Application-defined events. - `debug`: Detailed execution
information.

Example:

``` python
for event in graph.stream(
    {"messages": [HumanMessage(content="Explain graph state")]},
    stream_mode="updates",
):
    print(event)
```

Different modes produce different event shapes. Inspect the actual
returned structure and use the mode appropriate for debugging or
user-facing streaming.

# 19. Retries and Error Handling

## 19.1 Retry policies

LangGraph supports retry configuration for nodes. For example, a policy
configured with `max_attempts=3` generally means three total attempts
including the initial attempt. Exact retry behavior depends on the
installed version and policy.

Retries are appropriate for transient failures, such as temporary
network errors. They are not a substitute for correcting invalid schemas
or missing configuration.

## 19.2 Failure handling

Plan for: - Invalid model responses. - Tool failures. - Empty search
results. - API rate limits. - Timeouts. - Authentication errors. -
Incorrect state shapes. - Bad conditional routes. - Loops that do not
terminate.

Log the failing node and relevant error details without logging secrets
or unnecessary private content.

## 19.3 Side effects and retries

If a node sends an email, writes a database row, or performs another
irreversible action, retrying can repeat that action. Use idempotency
keys, deduplication, approval steps, or other safeguards as appropriate.

# 20. Human-in-the-Loop (HITL)

Human-in-the-loop workflows pause for human input, review, or approval.

Examples: - Review a generated response before publishing it. - Approve
a database update. - Confirm an email before sending. - Choose between
ambiguous actions. - Review a high-impact recommendation.

In LangGraph, `interrupt()` can pause execution, and
`Command(resume=...)` can supply a human response. A checkpointer and
the correct thread configuration are generally needed to resume the same
execution.

Important design considerations: 1. Show the human the relevant context.
2. Validate the supplied decision. 3. Pause before irreversible side
effects. 4. Handle timeout or abandonment. 5. Understand that a node may
execute again when resumed. 6. Make side-effecting actions idempotent
where possible.

# 21. RAG Workflows in LangGraph

RAG means Retrieval-Augmented Generation: retrieve relevant source
material and supply it as context to a model. Here the focus is on **how
to orchestrate RAG using LangGraph**, not on replacing the graph
concepts with a separate framework tutorial.

## 21.1 Basic RAG graph

``` text
START
  |
  v
retrieve_documents
  |
  v
build_context
  |
  v
generate_answer
  |
  v
END
```

A simple workflow has three responsibilities: - Retrieve relevant
documents. - Prepare context and source metadata. - Generate an answer
grounded in that context.

## 21.2 State schema

A graph might carry fields such as:

``` python
from typing_extensions import TypedDict

class RAGState(TypedDict):
    question: str
    documents: list
    context: str
    answer: str
```

This is a conceptual schema. In a production project, use the document
types and fields appropriate to the installed integrations.

## 21.3 Retrieval node

``` python
def retrieve_node(state: RAGState):
    docs = retriever.invoke(state["question"])
    return {"documents": docs}
```

This node calls the configured retriever and returns the retrieved
documents into graph state.

## 21.4 Context-building node

``` python
def context_node(state: RAGState):
    context = "\n\n".join(
        doc.page_content for doc in state["documents"]
    )
    return {"context": context}
```

A production version should also preserve source identifiers and
metadata, apply token limits, and handle the case where no documents are
retrieved.

## 21.5 Generation node

``` python
def generate_node(state: RAGState):
    prompt = f"""
    Answer using the provided context.
    If the context does not contain the answer, say so.
    Do not invent sources.

    Context:
    {state["context"]}

    Question:
    {state["question"]}
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}
```

The prompt is illustrative. Prefer an appropriate prompt template and
message structure in the actual application. Instructions alone cannot
guarantee grounding, so evaluate the output and citations.

## 21.6 Wire the graph

``` python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("prepare_context", context_node)
builder.add_node("generate", generate_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "prepare_context")
builder.add_edge("prepare_context", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()
```

The `retriever` and `llm` must be configured separately, and the corpus
must already be indexed.

## 21.7 Conditional RAG

A more advanced graph can assess whether retrieval succeeded:

``` text
START -> retrieve -> evaluate_retrieval
                         |
               +---------+---------+
               |                   |
          good evidence       weak evidence
               |                   |
               v                   v
           generate          rewrite query
               |                   |
               v                   v
              END              retrieve again
```

A real implementation needs: - A defined retrieval-quality field. - An
evaluator node that updates it. - A router that reads it. - A bounded
retry count. - A fallback when correction fails.

## 21.8 Debugging a RAG graph

Inspect each stage independently: 1. Does the source load correctly? 2.
Are expected chunks created? 3. Was the corpus indexed? 4. Does the
retriever return relevant documents? 5. Does the graph state contain
those documents? 6. Is context construction correct? 7. Does the model
see the intended context? 8. Does the answer make unsupported claims? 9.
Do citations map to the retrieved source metadata? 10. Does the graph
stop correctly when retrieval fails?

If the correct evidence is not retrieved, fix ingestion or retrieval
before trying to solve everything through prompt changes.

# 22. Self-RAG Workflow

Self-RAG incorporates self-reflection or critique signals into retrieval
and answer generation. Depending on the implementation, the system can
assess whether retrieval is needed, whether retrieved passages are
relevant, whether the answer is supported, and whether revision or
abstention is appropriate.

## 22.1 Conceptual graph

``` text
START
  |
  v
decide_retrieval_need
  |
  +---- retrieval unnecessary ----> generate
  |
  v
retrieve
  |
  v
evaluate_passages
  |
  +---- weak evidence ----> rewrite_query -> retrieve
  |
  v
generate_draft
  |
  v
evaluate_support
  |
  +---- unsupported ----> revise or retrieve again
  |
  v
END
```

## 22.2 Required graph state

An implementation may track: - `question` - `documents` -
`draft_answer` - `retrieval_quality` - `support_quality` -
`iterations` - `final_answer`

The fields alone do not create the behavior. Nodes must update them, and
conditional edges must route based on their values.

## 22.3 Important distinction

A graph with a retrieval node and an LLM-as-judge node can be a useful
self-checking RAG prototype. It should not automatically be described as
a faithful implementation of the original Self-RAG method unless the
relevant method-specific mechanisms are actually implemented.

## 22.4 Evaluation questions

-   Does the workflow know when retrieval is needed?
-   Does it assess passage relevance?
-   Does it check whether claims are supported?
-   Does it revise or abstain when evidence is weak?
-   Does it have a hard iteration limit?
-   Does it improve measured quality enough to justify extra latency and
    cost?

# 23. Corrective RAG (CRAG) Workflow

Corrective RAG focuses on assessing the quality of retrieved evidence
and correcting poor retrieval before generating the final answer.

## 23.1 Conceptual graph

``` text
START
  |
  v
initial_retrieval
  |
  v
evaluate_retrieval
  |
  +---- relevant ----> prepare_context
  |                         |
  |                         v
  |                    generate_answer
  |                         |
  |                         v
  |                        END
  |
  +---- weak/uncertain ----> correct_retrieval
                                |
                                v
                          filter_evidence
                                |
                                v
                          evaluate_again
                                |
                                v
                          generate or abstain
```

Correction may involve rewriting the query, retrieving from another
source, using web search, filtering irrelevant chunks, or combining
evidence.

## 23.2 State and routing

The graph needs a representation of retrieval quality, a correction
strategy, a retry counter, and a clear fallback. The routing logic
should not keep searching indefinitely.

## 23.3 Basic RAG vs Self-RAG vs CRAG

  -----------------------------------------------------------------------
  Workflow                            Primary emphasis
  ----------------------------------- -----------------------------------
  Basic RAG                           Retrieve and generate

  Self-RAG                            Reflection signals guide retrieval
                                      and assess evidence/output

  CRAG                                Evaluate and correct weak retrieval
  -----------------------------------------------------------------------

These are compatible ideas, not mutually exclusive categories.

# 24. GraphRAG and Agentic Workflows

## 24.1 GraphRAG

GraphRAG generally uses graph-oriented representations of entities and
relationships to support retrieval over connections or corpus-wide
themes. A LangGraph workflow can orchestrate graph retrieval, evidence
combination, and answer generation.

## 24.2 Agentic RAG

An agentic RAG workflow lets an agent choose retrieval actions
dynamically. For example, it may decide to: - Search a knowledge base. -
Rewrite the question. - Retrieve more evidence. - Call another approved
tool. - Stop and answer. - Admit that evidence is insufficient.

Use this complexity when it solves a real retrieval problem. Every
additional branch and loop needs a testable purpose and a safe stopping
condition.

# 25. Debugging Checklist

## 25.1 General graph debugging

-   [ ] Check the input state shape.
-   [ ] Check the graph state schema.
-   [ ] Confirm node names and registered callables.
-   [ ] Trace every edge from `START`.
-   [ ] Verify conditional-router return values.
-   [ ] Confirm each node returns valid state updates.
-   [ ] Check reducer behavior.
-   [ ] Confirm the expected node is actually reached.
-   [ ] Add logging/tracing around important nodes.
-   [ ] Verify the stop condition.
-   [ ] Test error and empty-result paths.
-   [ ] Add a hard limit for loops.

## 25.2 Agent and tool debugging

-   [ ] Inspect the `AIMessage`.
-   [ ] Check whether `tool_calls` is populated.
-   [ ] Confirm the model is configured with the intended tool
    definitions.
-   [ ] Verify tool names match exactly.
-   [ ] Verify tool-call arguments match the registered input schema.
-   [ ] Confirm the `ToolNode` is registered in the graph.
-   [ ] Confirm the graph edge reaches the tool node.
-   [ ] Inspect `ToolMessage` results.
-   [ ] Verify results reach the next model call.
-   [ ] Test the no-tool/final-answer path.
-   [ ] Test empty and failed tool results.
-   [ ] Verify the final output shape.

## 25.3 Reflection/Reflexion debugging

-   [ ] Confirm the initial draft is added to message state.
-   [ ] Confirm critique/revision sees the intended history.
-   [ ] Confirm the critique is represented appropriately.
-   [ ] Verify tool schemas against function signatures.
-   [ ] Ensure the revision chain knows how to use tool results.
-   [ ] Count actual iterations rather than unrelated messages.
-   [ ] Verify the maximum iteration boundary.
-   [ ] Check source URLs and evidence quality.
-   [ ] Decide how incomplete or unsupported answers should end.
-   [ ] If claiming Reflexion memory, verify lessons are actually stored
    and reused.

## 25.4 RAG graph debugging

-   [ ] Confirm ingestion is correct.
-   [ ] Inspect chunks.
-   [ ] Verify embedding/index configuration.
-   [ ] Test retrieval independently.
-   [ ] Inspect documents in graph state.
-   [ ] Inspect the assembled context.
-   [ ] Evaluate grounding and citations.
-   [ ] Test questions with absent answers.
-   [ ] Bound correction loops.
-   [ ] Compare quality and latency before/after adding complexity.

# 26. KT and Interview Revision

## 26.1 Short definitions

**LangGraph:** A graph-based framework for stateful workflows with
nodes, edges, branching, loops, and optional persistence.

**State:** The data passed through graph execution.

**Node:** A callable/runnable that reads state and returns updates.

**Edge:** A transition from one graph node to another.

**Conditional edge:** A transition selected by routing logic.

**Reducer:** A function defining how a new state update combines with
the existing value.

**`MessagesState`:** A predefined state schema for message-oriented
workflows.

**`add_messages`:** A reducer that applies message-aware update
behavior.

**`compile()`:** Prepares a graph as an executable runnable.

**`invoke()`:** Executes a compiled graph for an input.

**Checkpointing:** Saving graph execution state for continuation or
recovery.

**Thread ID:** Identifier associated with a checkpointed workflow
thread.

**ReAct:** An agent pattern alternating between decisions, actions, and
observations.

**Reflection:** Generate, critique, and revise an output.

**Reflexion:** Use evaluated attempts and verbal lessons to improve
later attempts, often through memory.

**`ToolNode`:** A prebuilt node that executes model-requested tool
calls.

**RAG graph:** A graph that orchestrates retrieval, context preparation,
and grounded generation.

**Self-RAG:** RAG approach that incorporates self-reflection signals
into retrieval and output assessment.

**CRAG:** RAG approach that evaluates retrieval quality and corrects
weak evidence.

## 26.2 Common interview questions

### What is the difference between LangGraph and a simple chain?

A simple chain often executes a predetermined sequence. LangGraph
explicitly models state, nodes, edges, branching, loops, and optional
checkpointing, making it useful for more complex stateful workflows.

### What is the difference between `add_edge` and `add_conditional_edges`?

`add_edge` defines a fixed transition. `add_conditional_edges` chooses a
destination based on a routing function or condition.

### Does `compile()` run the graph?

No. It prepares the executable graph. `invoke()` runs it.

### What is the difference between an AI tool call and a ToolMessage?

An `AIMessage` may request tool execution. A `ToolMessage` contains the
result after a tool executes.

### Does `bind_tools()` execute tools?

No. It normally exposes tool definitions to the model. The application
or a node such as `ToolNode` performs execution.

### Why use a reducer?

To control how new state updates combine with existing values---for
example, appending messages instead of replacing the entire list.

### Does `MessagesState` automatically provide long-term memory?

No. It provides message-oriented state. Persistence requires a
checkpointer; long-term cross-thread memory needs additional design.

### Why add a maximum iteration count?

To prevent infinite loops and bound cost/latency, even when the
evaluator or model fails to terminate.

### What is the difference between Reflection and Reflexion?

Reflection critiques and improves the current output. Reflexion
emphasizes lessons from previous attempts that can guide later attempts.

### What is the role of a router?

It chooses the next graph destination based on state or another
condition.

### How would you implement RAG in LangGraph?

Define state for the question, documents, context, and answer; create
retrieval, context-building, and generation nodes; connect them with
edges; optionally add evaluation/correction branches and a bounded retry
loop.

### How would you debug an agent that repeatedly calls a tool?

Inspect tool-call messages, tool schemas, tool results, routing logic,
state updates, and the iteration counter. Add a hard limit and test
empty/failure cases.

## 26.3 Explaining a graph in a KT

For any graph, explain it in this order: 1. What problem does it solve?
2. What fields are in the state? 3. What nodes are registered? 4. What
does each node read and return? 5. Where does execution start? 6. Which
edges are fixed? 7. Which edges are conditional? 8. How does state
change between nodes? 9. What makes the graph stop? 10. How are failures
handled? 11. What are the limitations and improvements?

# 27. Current Status and Next Steps

## Completed

-   [x] LangGraph graph fundamentals: state, nodes, edges, `START`,
    `END`.
-   [x] `compile()` versus `invoke()`.
-   [x] Partial state updates and reducers.
-   [x] `MessagesState` and message-aware state updates.
-   [x] Fixed edges and conditional routing.
-   [x] Looping and stopping conditions.
-   [x] ReAct agent.
-   [x] Tool calling and `ToolNode`.
-   [x] Reflection agent.
-   [x] Reflexion-style research-and-revision agent.
-   [x] Review of the Reflexion graph's stopping condition and potential
    tool-schema mismatch.
-   [x] Concepts of checkpointing, memory, streaming, retries, and
    human-in-the-loop.

## Current topic: RAG with LangGraph

-   [ ] Define the graph state.
-   [ ] Implement retrieval node.
-   [ ] Inspect retrieved documents.
-   [ ] Implement context-building node.
-   [ ] Implement grounded generation node.
-   [ ] Connect the nodes and invoke the graph.
-   [ ] Test an answerable question.
-   [ ] Test a question absent from the corpus.
-   [ ] Add evaluation and bounded correction only after the basic graph
    works.

## Next

1.  Self-RAG workflow.
2.  Corrective RAG (CRAG) workflow.
3.  Compare the agent and RAG patterns.
4.  Final LangGraph KT revision, including diagrams, code tracing, and
    interview questions.

# 28. Glossary

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  Agent                               Controller that can choose actions
                                      based on model output and
                                      observations

  `AIMessage`                         Model message that may contain a
                                      tool-call request

  Checkpointer                        Component that saves graph
                                      execution state

  Conditional route                   Branch selected by a
                                      router/condition

  `END`                               Special terminal destination

  Graph                               Nodes and edges describing workflow
                                      execution

  `HumanMessage`                      User or human-authored message

  HITL                                Human-in-the-loop review or
                                      approval

  Node                                Unit of graph work

  Reducer                             Function that combines state
                                      updates

  `START`                             Special graph entry point

  State                               Data shared across graph execution

  `StateGraph`                        Graph builder for a defined state
                                      schema

  Thread ID                           Identifier for checkpointed thread
                                      state

  `ToolMessage`                       Tool execution result message

  `ToolNode`                          Prebuilt tool execution node

  `MessagesState`                     Predefined message-oriented state
                                      schema

  ReAct                               Reasoning-and-acting agent pattern

  Reflection                          Critique and revise the current
                                      output

  Reflexion                           Learn from past attempts through
                                      reusable lessons

  RAG                                 Retrieval-Augmented Generation

  Self-RAG                            RAG with self-reflection/critique
                                      signals

  CRAG                                Corrective Retrieval-Augmented
                                      Generation

  Iteration limit                     Hard bound on repeated graph cycles
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Final notes

This document is intentionally **LangGraph-only**. RAG, Self-RAG, and
CRAG are included only in the context of workflows orchestrated with
LangGraph. It does not attempt to serve as a general LangChain
reference.

The two most important implementation checks from the Reflexion-style
project are: 1. The registered tool input schemas must match the
`run_queries(search_queries, **kwargs)` function signature and the
model's actual tool-call arguments. 2. The current loop condition counts
`ToolMessage` objects, not actual iteration cycles. A dedicated counter
is more reliable.


---

# Adaptive RAG with LangGraph

## 1. What is Adaptive RAG?

**Adaptive RAG (Adaptive Retrieval-Augmented Generation)** selects an
appropriate answering or retrieval strategy based on the user's question
and the available evidence. Instead of sending every question through
the same retrieval pipeline, the graph can route different questions to
different workflows.

For example:

-   A simple question may be answered directly without retrieval.
-   A question about private project documentation may use a local
    retriever.
-   A question requiring current information may use web search.
-   A complex question may use retrieval plus additional evaluation or
    correction.

Adaptive RAG is a broad design approach rather than one mandatory graph
structure. The exact routing logic depends on the application.

## 2. Why do we need Adaptive RAG?

A single fixed RAG pipeline can waste resources or choose the wrong
information source:

1.  Retrieving documents for every simple question can add unnecessary
    latency and cost.
2.  Using only a local vector database may fail when the question asks
    about current information that is not in the database.
3.  Using web search for every question is unnecessary when the answer
    is already available in trusted internal documents.
4.  Complex questions may need a stronger workflow than simple
    questions.

Adaptive RAG addresses these problems by choosing a route suited to the
question.

## 3. Example workflow

A common LangGraph implementation can use this structure:

``` text
                         START
                           |
                           v
                    classify_question
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        direct_answer  local_retrieval  web_search
             |             |             |
             |             v             |
             |       grade_documents     |
             |             |             |
             |       +-----+-----+       |
             |       |           |       |
             |       v           v       |
             |   generate    web_search  |
             |       |           |       |
             +-------+-----------+-------+
                         |
                         v
                   final_answer
                         |
                        END
```

This is an illustrative architecture, not a required Adaptive RAG
design. A real graph may merge nodes, use a different classifier, or
include answer evaluation and retries.

## 4. How it differs from CRAG and Self-RAG

  -----------------------------------------------------------------------
  Approach                Main decision or check  Typical behavior
  ----------------------- ----------------------- -----------------------
  Basic RAG               No explicit adaptive    Retrieve, then generate
                          decision                

  CRAG                    Are retrieved documents Use good local evidence
                          relevant?               or correct retrieval,
                                                  for example with web
                                                  search

  Self-RAG                Should retrieval be     Reflect on retrieval
                          used, and is the        and generation
                          evidence-supported      
                          answer good?            

  Adaptive RAG            Which                   Route to direct
                          answering/retrieval     answering, local
                          strategy suits this     retrieval, web search,
                          question?               or another workflow
  -----------------------------------------------------------------------

These approaches overlap and can be combined. For example, an Adaptive
RAG router can choose a CRAG workflow for a question that needs local
documents, and that workflow can be extended with answer evaluation
inspired by Self-RAG.

**Memory aid:** - CRAG corrects retrieval. - Self-RAG reflects on
retrieval and generation. - Adaptive RAG chooses the strategy.

## 5. Define the graph state

LangGraph nodes read the shared state and return dictionaries containing
the fields they want to update. The following example uses a `TypedDict`
state with fields for the question, route, retrieved documents, context,
and answer.

``` python
from typing import Any, TypedDict

class GraphState(TypedDict, total=False):
    question: str
    route: str
    documents: list[Any]
    context: str
    answer: str
```

`total=False` makes the keys optional in the type definition, which is
useful because not every route populates every field. In a production
graph, define more precise document and output types where practical.

## 6. Build a router

The router decides which route should handle the question. A simple
implementation can use rules, while a more flexible implementation can
use an LLM classifier with structured output.

### Option A: Rule-based router

``` python
def route_question(state: GraphState) -> str:
    question = state["question"].lower()

    if any(term in question for term in [
        "latest", "today", "current", "recent", "this week"
    ]):
        return "web_search"

    if any(term in question for term in [
        "my documents", "project documentation", "internal docs"
    ]):
        return "local_retrieval"

    return "direct_answer"
```

This is easy to understand, but keyword routing is limited. It may
misclassify questions when users phrase them differently. For a real
application, use an LLM-based classifier or combine rules with a
classifier.

### Option B: LLM-based router

An LLM router can return a constrained label such as `direct_answer`,
`local_retrieval`, or `web_search`. Use a schema with an enum or
`Literal` type so the output is constrained to valid routes, then
validate it before routing.

``` python
from typing import Literal
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    route: Literal[
        "direct_answer",
        "local_retrieval",
        "web_search",
    ] = Field(
        description="Choose the best strategy for answering the question."
    )
```

The router model should be given clear criteria: - `direct_answer`: a
stable, simple question that does not need external evidence. -
`local_retrieval`: a question likely to be answered by the application's
trusted document collection. - `web_search`: a question requiring
current or externally available information.

The router's choice is a decision, not proof that the chosen source
contains the answer. Later nodes should still handle missing or
low-quality evidence.

## 7. Implement the nodes

The following is a framework for the workflow. Replace the placeholder
functions with your actual LLM, vector-store retriever, and web-search
tool.

``` python
def direct_answer(state: GraphState) -> GraphState:
    question = state["question"]

    # Replace with your LLM call.
    answer = answer_without_retrieval(question)

    return {"answer": answer}


def retrieve_local(state: GraphState) -> GraphState:
    question = state["question"]

    # Replace with your vector-store retriever.
    documents = retriever.invoke(question)

    return {"documents": documents}


def grade_documents(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state.get("documents", [])

    # Replace with your relevance-grading chain.
    relevant_documents = grade_and_filter_documents(
        question, documents
    )

    return {"documents": relevant_documents}


def web_search(state: GraphState) -> GraphState:
    question = state["question"]

    # Replace with your web-search tool.
    web_documents = search_web(question)

    # This example replaces the document list with web results.
    # If both local and web evidence are needed, combine and label them.
    return {"documents": web_documents}


def generate_answer(state: GraphState) -> GraphState:
    question = state["question"]
    documents = state.get("documents", [])

    # Format documents into context before calling the LLM.
    context = "\n\n".join(
        document.page_content for document in documents
    )

    # Replace with your grounded answer-generation chain.
    answer = generate_from_context(question, context)

    return {
        "context": context,
        "answer": answer,
    }
```

The functions `answer_without_retrieval`, `grade_and_filter_documents`,
`search_web`, and `generate_from_context` are placeholders, not built-in
LangGraph functions. They must be implemented using the chosen model and
tools.

## 8. Connect the graph with conditional edges

``` python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(GraphState)

builder.add_node("direct_answer", direct_answer)
builder.add_node("retrieve_local", retrieve_local)
builder.add_node("grade_documents", grade_documents)
builder.add_node("web_search", web_search)
builder.add_node("generate_answer", generate_answer)

builder.add_conditional_edges(
    START,
    route_question,
    {
        "direct_answer": "direct_answer",
        "local_retrieval": "retrieve_local",
        "web_search": "web_search",
    },
)

builder.add_edge("direct_answer", END)

builder.add_edge("retrieve_local", "grade_documents")

def route_after_grading(state: GraphState) -> str:
    if state.get("documents"):
        return "generate_answer"
    return "web_search"

builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate_answer": "generate_answer",
        "web_search": "web_search",
    },
)

builder.add_edge("web_search", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
```

Invoke the graph with a question:

``` python
result = graph.invoke({
    "question": "What is the latest LangGraph release?"
})

print(result["answer"])
```

Important implementation notes:

-   This code is a structural template. The placeholder model and search
    functions must exist before it can run.
-   The simple post-grading route above treats a non-empty document list
    as usable. In a real CRAG-style workflow, use the relevance grader's
    explicit result or filtered documents to decide whether local
    evidence is sufficient.
-   If local documents are irrelevant, the workflow should not generate
    an answer as if they were valid evidence.
-   If the web search returns no useful evidence, handle that case
    explicitly rather than forcing an unsupported answer.
-   The direct-answer path can still benefit from confidence checks,
    citations, or a policy requiring retrieval for high-stakes
    questions.
-   Add retry limits to any loop so a question cannot trigger endless
    retrieval or generation attempts.

## 9. A stronger Adaptive RAG design using CRAG

The workflow above can be improved by making the local-retrieval path
use the CRAG logic you already learned:

``` text
START
  |
  v
route_question
  |
  +--> direct_answer ----------------------> END
  |
  +--> local_retrieval --> grade_documents
  |                            |
  |                    +-------+-------+
  |                    |               |
  |               relevant         irrelevant
  |                    |               |
  |                    v               v
  |                generate        web_search
  |                    |               |
  |                    |               v
  |                    |            generate
  |                    |               |
  +--> web_search -----+---------------+
                                       |
                                       v
                                      END
```

This combines two ideas: 1. **Adaptive routing** chooses the initial
strategy. 2. **Corrective retrieval** checks whether the local retrieval
path needs web-search fallback.

An additional answer-grading node can be added after generation to check
support and usefulness, creating a Self-RAG-inspired reflection step.

## 10. Where Self-RAG fits

Adaptive RAG, CRAG, and Self-RAG do not have to be separate
applications.

A combined workflow can: 1. Route the question to a suitable strategy
(Adaptive RAG). 2. Grade local retrieval and fall back to web search if
needed (CRAG). 3. Evaluate whether the final answer is supported and
useful, then revise or retry when appropriate (Self-RAG-inspired
behavior).

The exact original Self-RAG method uses a model trained to produce
special reflection tokens; an ordinary LangGraph loop with an LLM
evaluator is an inspired engineering implementation, not an exact
reproduction of that research method.

## 11. Testing Adaptive RAG

Test each route separately:

  -----------------------------------------------------------------------
  Test question           Expected route          What to verify
  ----------------------- ----------------------- -----------------------
  "What is 2 + 2?"        Direct answer           No unnecessary
                                                  retrieval

  "Explain the            Local retrieval         Relevant local passages
  `StateGraph` class in                           are retrieved
  my documentation."                              

  "What is the latest     Web search              Search is used for
  LangGraph release?"                             up-to-date evidence

  A question whose local  Local retrieval, then   CRAG correction
  documents are           web fallback            activates
  irrelevant                                      

  A question for which    Appropriate retrieval   The system does not
  neither source has      path, then graceful     invent an answer
  evidence                fallback                
  -----------------------------------------------------------------------

Track route accuracy, retrieval relevance, answer correctness,
groundedness, latency, and cost. Routing that looks sensible in a few
examples should still be evaluated on a representative set of real
questions.

## 12. Final comparison

  --------------------------------------------------------------------------
  Feature        Basic RAG      CRAG           Self-RAG       Adaptive RAG
  -------------- -------------- -------------- -------------- --------------
  Retrieves      Yes            Yes            When useful    Depending on
  documents                                                   route

  Checks         Not            Central        Yes, as part   Depends on
  retrieval      necessarily    feature        of reflection  selected
  relevance                                                   workflow

  Uses web       Not            Common         Possible       One available
  fallback       necessarily    corrective                    strategy
                                action                        

  Evaluates the  Not            Not its        Central        Optional,
  final answer   necessarily    central focus  feature        depending on
                                                              design

  Chooses        Usually no     Not its main   Retrieval      Central
  between                       focus          decision is    feature
  multiple                                     part of its    
  initial                                      method         
  strategies                                                  
  --------------------------------------------------------------------------

**Takeaway:** Adaptive RAG chooses the strategy, CRAG corrects poor
retrieval, and Self-RAG reflects on retrieval and generation. In
LangGraph, these ideas can be represented through state, nodes,
conditional edges, and bounded retry loops.
