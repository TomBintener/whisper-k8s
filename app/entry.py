#!/usr/bin/env python3
"""
Single entrypoint for the whisper-suite image.

Chooses which component to run based on SERVICE env:

- SERVICE=bridge      -> run FastAPI bridge via uvicorn
- SERVICE=worker      -> run video_transcriber worker once and exit
- SERVICE=dispatcher  -> run dispatcher batch scheduler
"""

import os
import sys
import logging

import uvicorn


def main() -> int:
    service = os.getenv("SERVICE", "bridge").strip().lower()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if service == "worker":
        # Only import worker when needed so bridge does not depend on worker env
        import video_transcriber  # type: ignore

        return video_transcriber.main()

    if service == "dispatcher":
        # Only import dispatcher when needed so bridge does not require BRIDGE_JOB_ID
        import dispatcher  # type: ignore

        dispatcher.main()
        return 0

    # Default: run the HTTP bridge
    import bridge  # type: ignore

    port = int(os.getenv("PORT", "8080"))
    logging.getLogger(__name__).info(
        "Starting bridge on 0.0.0.0:%d (SERVICE=%s)", port, service
    )
    uvicorn.run(
        bridge.app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
