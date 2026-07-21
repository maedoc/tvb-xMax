"""tvb-max CLI."""
import argparse


def main():
    p = argparse.ArgumentParser(prog="tvbmax",
                               description="Advanced AI math compiler for virtual brain simulation")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("compile", help="compile a spec into an artifact")
    sub.add_parser("infer", help="run a compiled artifact (fast path)")
    sub.add_parser("swap", help="apply a free swap and re-run")
    sub.add_parser("serve", help="start the FastAPI server")
    sub.add_parser("agents", help="run the openclaw discord agents")
    sub.add_parser("leaderboard", help="print the current leaderboard")

    args = p.parse_args()
    if args.cmd == "serve":
        import uvicorn
        from tvb_max.api import app
        uvicorn.run(app, host="0.0.0.0", port=8088)
    elif args.cmd == "agents":
        from tvb_max.community import run_bot
        import os
        token = os.environ.get("TVBMAX_DISCORD_TOKEN", "")
        from tvb_max.community.discord_bot import OpenClawAgent
        agents = [
            OpenClawAgent(name="openclaw-hopf", model="hopf", channel_id=0),
            OpenClawAgent(name="openclaw-mpr", model="mpr", channel_id=0),
        ]
        run_bot(token, agents)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
