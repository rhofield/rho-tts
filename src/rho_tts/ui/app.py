"""
Gradio Blocks application for the rho-tts web UI.

Three-tab layout:
  1. Generate — model/voice selection, text input, audio playback, phonetic mappings
  2. Voices  — manage voice profiles (upload reference audio, set transcript)
  3. Models  — manage model configurations (provider params, thresholds)
"""

import argparse
import logging
import os
from typing import Optional

import gradio as gr

from . import callbacks
from .config import PROVIDER_MODELS, get_provider_model_choices, load_config
from .state import AppState

logger = logging.getLogger(__name__)


def _build_app(state: AppState) -> gr.Blocks:
    """Construct the Gradio Blocks UI and wire all events."""

    with gr.Blocks(title="rho-tts Studio", theme=gr.themes.Soft()) as app:

        # ── shared helpers ──────────────────────────────────────────

        def _voice_choices():
            return gr.Dropdown(choices=callbacks.voice_choices(state.config))

        def _model_choices():
            return gr.Dropdown(choices=callbacks.model_choices(state.config))

        def _refresh_dropdowns():
            return _model_choices(), _voice_choices()

        # ── Tab 1: Generate ─────────────────────────────────────────

        with gr.Tab("Generate"):
            with gr.Row():
                gr.Markdown("## rho-tts Studio")
                device_dd = gr.Dropdown(
                    choices=["cuda", "cpu"],
                    value=state.config.device,
                    label="Device",
                    scale=0,
                    min_width=120,
                )

            with gr.Row():
                # -- Config sidebar --
                with gr.Column(scale=1, min_width=260):
                    model_dd = gr.Dropdown(
                        choices=callbacks.model_choices(state.config),
                        label="Model",
                        interactive=True,
                    )
                    voice_dd = gr.Dropdown(
                        choices=callbacks.voice_choices(state.config),
                        label="Voice",
                        interactive=True,
                    )

                # -- Generation area --
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        placeholder="Enter text to synthesize...",
                        lines=5,
                    )
                    with gr.Row():
                        gen_btn = gr.Button("Generate", variant="primary")
                        cancel_btn = gr.Button("Cancel", variant="stop")
                    status_box = gr.Textbox(label="Status", interactive=False)
                    audio_out = gr.Audio(label="Generated Audio", type="filepath")

            # -- Phonetic mapping accordion --
            with gr.Accordion("Phonetic Mapping", open=False):
                mapping_label = gr.Markdown("Select a voice and model to load mappings.")
                mapping_df = gr.Dataframe(
                    headers=["Word", "Pronunciation"],
                    col_count=(2, "fixed"),
                    interactive=True,
                    row_count=(3, "dynamic"),
                    label="Phonetic Mappings",
                )
                with gr.Row():
                    save_mapping_btn = gr.Button("Save Mappings")
                    mapping_status = gr.Textbox(label="", interactive=False, show_label=False)

            # -- Generate tab events --

            def _on_device_change(device):
                state.config.device = device
                state.save()
                state.invalidate_tts()

            device_dd.change(fn=_on_device_change, inputs=[device_dd])

            def _on_voice_or_model_change(voice_id, model_id):
                """Reload phonetic mapping when voice or model changes."""
                rows, label = callbacks.load_phonetic_mapping(state, voice_id, model_id)
                return rows, label

            voice_dd.change(
                fn=_on_voice_or_model_change,
                inputs=[voice_dd, model_dd],
                outputs=[mapping_df, mapping_label],
            )
            model_dd.change(
                fn=_on_voice_or_model_change,
                inputs=[voice_dd, model_dd],
                outputs=[mapping_df, mapping_label],
            )

            def _generate(model_id, voice_id, text):
                path, msg = callbacks.generate_audio(state, model_id, voice_id, text)
                return path, msg

            gen_btn.click(
                fn=_generate,
                inputs=[model_dd, voice_dd, text_input],
                outputs=[audio_out, status_box],
                concurrency_limit=1,
            )

            cancel_btn.click(
                fn=lambda: callbacks.cancel_generation(state),
                outputs=[status_box],
                concurrency_limit=None,
            )

            def _save_mapping(voice_id, model_id, df_data):
                rows = df_data.values.tolist() if hasattr(df_data, "values") else df_data
                return callbacks.save_phonetic_mapping(state, voice_id, model_id, rows)

            save_mapping_btn.click(
                fn=_save_mapping,
                inputs=[voice_dd, model_dd, mapping_df],
                outputs=[mapping_status],
            )

        # ── Tab 2: Voices ───────────────────────────────────────────

        with gr.Tab("Voices"):
            gr.Markdown("### Manage Voice Profiles")
            with gr.Row():
                v_id = gr.Textbox(label="Voice ID", placeholder="narrator_ryan")
                v_name = gr.Textbox(label="Display Name", placeholder="Ryan - Narrator")
            v_audio = gr.Audio(label="Reference Audio", type="filepath")
            v_ref_text = gr.Textbox(
                label="Reference Text",
                placeholder="Transcript of the reference audio (required for Qwen)",
                lines=2,
            )
            v_add_btn = gr.Button("Add / Update Voice", variant="primary")
            v_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Saved Voices")
            v_table = gr.Dataframe(
                value=callbacks._voices_table(state.config),
                headers=["ID", "Name", "Audio Path", "Reference Text"],
                interactive=False,
                label="Voice Profiles",
            )
            with gr.Row():
                v_del_id = gr.Textbox(label="Voice ID to Delete", scale=2)
                v_del_btn = gr.Button("Delete Voice", variant="stop", scale=0, min_width=140)

            # -- Voice events --

            def _add_voice(vid, vname, vaudio, vref):
                table, msg = callbacks.add_voice(state, vid, vname, vaudio, vref)
                return table, msg, _voice_choices(), _model_choices()

            v_add_btn.click(
                fn=_add_voice,
                inputs=[v_id, v_name, v_audio, v_ref_text],
                outputs=[v_table, v_status, voice_dd, model_dd],
            )

            def _del_voice(vid):
                table, msg = callbacks.delete_voice(state, vid)
                return table, msg, _voice_choices(), _model_choices()

            v_del_btn.click(
                fn=_del_voice,
                inputs=[v_del_id],
                outputs=[v_table, v_status, voice_dd, model_dd],
            )

        # ── Tab 3: Models ───────────────────────────────────────────

        with gr.Tab("Models"):
            gr.Markdown("### Add Model")
            m_provider = gr.Dropdown(
                choices=list(PROVIDER_MODELS.keys()),
                label="Provider",
                interactive=True,
            )
            m_model_select = gr.Dropdown(
                choices=[],
                label="Model",
                interactive=True,
            )

            with gr.Accordion("Parameters", open=False):
                m_max_iter = gr.Number(label="max_iterations", value=10, precision=0)
                m_accent = gr.Slider(
                    label="accent_drift_threshold",
                    minimum=0, maximum=1, step=0.01, value=0.17,
                )
                m_text_sim = gr.Slider(
                    label="text_similarity_threshold",
                    minimum=0, maximum=1, step=0.01, value=0.85,
                )

            m_add_btn = gr.Button("Add Model", variant="primary")
            m_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Saved Models")
            m_table = gr.Dataframe(
                value=callbacks._models_table(state.config),
                headers=["Name", "Provider", "Params"],
                interactive=False,
                label="Model Configurations",
            )
            with gr.Row():
                m_del_dd = gr.Dropdown(
                    choices=callbacks.model_choices(state.config),
                    label="Model to Delete",
                    interactive=True,
                    scale=2,
                )
                m_del_btn = gr.Button("Delete Model", variant="stop", scale=0, min_width=140)

            # -- Provider → Model cascade --

            def _on_provider_change(provider):
                choices = get_provider_model_choices(provider) if provider else []
                logger.info("Provider changed to %r, model choices: %s", provider, choices)
                return gr.Dropdown(choices=choices, value=None, interactive=True)

            m_provider.change(
                fn=_on_provider_change,
                inputs=[m_provider],
                outputs=[m_model_select],
            )

            def _on_model_select_change(provider, model_name):
                """Fill threshold defaults when a catalog model is selected."""
                if not provider or not model_name:
                    return gr.skip(), gr.skip(), gr.skip()
                from .config import get_provider_model_defaults
                defaults = get_provider_model_defaults(provider, model_name)
                return (
                    defaults.get("max_iterations", 10),
                    defaults.get("accent_drift_threshold", 0.17),
                    defaults.get("text_similarity_threshold", 0.85),
                )

            m_model_select.change(
                fn=_on_model_select_change,
                inputs=[m_provider, m_model_select],
                outputs=[m_max_iter, m_accent, m_text_sim],
            )

            # -- Model events --

            def _add_model(mprov, mmodel, miter, macc, mtsim):
                table, msg = callbacks.add_model(
                    state, mprov, mmodel, miter, macc, mtsim,
                )
                return (
                    table,
                    msg,
                    _model_choices(),
                    _voice_choices(),
                    gr.Dropdown(choices=callbacks.model_choices(state.config)),
                )

            m_add_btn.click(
                fn=_add_model,
                inputs=[m_provider, m_model_select, m_max_iter, m_accent, m_text_sim],
                outputs=[m_table, m_status, model_dd, voice_dd, m_del_dd],
            )

            def _del_model(model_id):
                table, msg = callbacks.delete_model(state, model_id)
                return (
                    table,
                    msg,
                    _model_choices(),
                    _voice_choices(),
                    gr.Dropdown(choices=callbacks.model_choices(state.config)),
                )

            m_del_btn.click(
                fn=_del_model,
                inputs=[m_del_dd],
                outputs=[m_table, m_status, model_dd, voice_dd, m_del_dd],
            )

    return app


def launch_ui(
    config_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    device: Optional[str] = None,
) -> None:
    """
    Build and launch the Gradio web UI.

    Can be called programmatically or via CLI (``rho-tts-ui``).
    """
    # Parse CLI args if invoked from the command line
    parser = argparse.ArgumentParser(description="rho-tts Web UI")
    parser.add_argument("--config", default=config_path, help="Path to config JSON file")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--share", action="store_true", default=share)
    parser.add_argument("--device", default=device, choices=["cuda", "cpu"])
    args, _ = parser.parse_known_args()

    config = load_config(args.config)
    if args.device:
        config.device = args.device

    state = AppState(config=config, config_path=args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    app = _build_app(state)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
