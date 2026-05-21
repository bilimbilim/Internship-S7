from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import asyncio
import uvicorn

from main_agent import build_agent_graph
from state import AgentState

app = FastAPI()
templates = Jinja2Templates(directory="templates")

agent = build_agent_graph()



# PAGE VISUALIZER

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("visualizer.html", {"request": request})


# WEBSOCKET LIVE STREAM

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            message = await websocket.receive_json()

            # Quand on clique sur "run"
            if message.get("action") == "run":

                async for step in agent.astream(AgentState()):
                    for node_name in step.keys():
                        await websocket.send_json({"node": node_name})
                        await asyncio.sleep(0.7)

                await websocket.send_json({"node": "END"})

        except Exception as e:
            await websocket.send_json({"error": str(e)})
            await websocket.close()
            break


if __name__ == "__main__":
    uvicorn.run("visualizer_server:app", host="127.0.0.1", port=9000, reload=True)
