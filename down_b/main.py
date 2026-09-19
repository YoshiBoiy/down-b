from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from down_b.api.websocket import WebSocketHub, run_server
from down_b.config import load_opponent_config, load_thresholds
from down_b.game import GameEngine
from down_b.simulation.body_stream import simulated_body_states
from down_b.storage.session_writer import SessionWriter
from down_b.types import ExchangeResult


async def _run(args: argparse.Namespace) -> None:
    thresholds = load_thresholds(args.thresholds)
    opponent = load_opponent_config(args.opponent)
    engine = GameEngine(thresholds, opponent)
    hub = WebSocketHub()
    writer = SessionWriter(args.sessions)
    server_task = asyncio.create_task(run_server(hub, args.web_root, args.host, args.port))
    print(f"Down-B simulator running at http://{args.host}:{args.port}")
    try:
        for state in simulated_body_states(args.fps):
            frame = engine.update(state)
            message = frame.to_json()
            message["body"] = state.to_json()
            writer.observe_frame(message)
            if frame.exchange_result in {ExchangeResult.HIT, ExchangeResult.BLOCK}:
                writer.record_event(frame.exchange_result.value, message)
            await hub.publish(message)
            await asyncio.sleep(1 / args.fps)
    except asyncio.CancelledError:
        raise
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Down-B simulator and browser visualization.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--web-root", default=str(Path("web")))
    parser.add_argument("--thresholds", default="config/thresholds.yaml")
    parser.add_argument("--opponent", default="config/opponent.yaml")
    parser.add_argument("--sessions", default="sessions")
    try:
        asyncio.run(_run(parser.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
