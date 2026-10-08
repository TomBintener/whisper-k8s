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

def preload_models(models: list[str]) -> int:
    """Pre-download and cache models on persistent storage to eliminate runtime cold starts."""
    logger = logging.getLogger("model-preloader")
    models_dir = os.getenv("MODELS_DIR", "/data/models")
    whisper_root = os.getenv("WHISPER_DOWNLOAD_ROOT", f"{models_dir}/whisper")
    hf_root = os.getenv("HF_HOME", f"{models_dir}/huggingface")

    os.makedirs(whisper_root, exist_ok=True)
    os.makedirs(hf_root, exist_ok=True)

    logger.info("Starting model preloading into %s for models: %s", models_dir, models)
    success = True

    for model_name in models:
        m = model_name.strip()
        if not m:
            continue
        logger.info("Pre-warming model '%s'...", m)

        # 1. Warm OpenAI Whisper weights
        try:
            import whisper
            logger.info("Downloading OpenAI Whisper weights for '%s' to %s", m, whisper_root)
            whisper.load_model(m, device="cpu", download_root=whisper_root)
            logger.info("Successfully primed OpenAI Whisper model '%s'", m)
        except Exception as e:
            logger.warning("Could not pre-warm OpenAI Whisper model '%s': %s", m, e)
            success = False

        # 2. Warm Faster-Whisper weights
        try:
            from faster_whisper import WhisperModel
            logger.info("Downloading Faster-Whisper weights for '%s' to %s", m, hf_root)
            WhisperModel(m, device="cpu", compute_type="float32", download_root=hf_root)
            logger.info("Successfully primed Faster-Whisper model '%s'", m)
        except Exception as e:
            logger.warning("Could not pre-warm Faster-Whisper model '%s': %s", m, e)

    if success:
        logger.info("All requested models preloaded successfully!")
        return 0
    logger.warning("One or more models encountered warnings during preloading.")
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Check for CLI argument: --preload base,small
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg in ("--preload", "-p"):
            models_to_preload = sys.argv[i + 1].split(",") if i < len(sys.argv) - 1 else ["base"]
            return preload_models(models_to_preload)

    service = os.getenv("SERVICE", "bridge").strip().lower()

    if service in ("preload", "warmup"):
        models_str = os.getenv("PRELOAD_MODELS", "base,small")
        models_to_preload = [m.strip() for m in models_str.split(",") if m.strip()]
        return preload_models(models_to_preload)

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
    import uvicorn
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
