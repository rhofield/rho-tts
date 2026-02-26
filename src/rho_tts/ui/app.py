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
from .config import PROVIDER_MODELS, get_provider_model_choices, get_provider_model_defaults, is_model_cached, load_config
from .state import AppState

logger = logging.getLogger(__name__)


def _build_app(state: AppState) -> gr.Blocks:
    """Construct the Gradio Blocks UI and wire all events."""

    with gr.Blocks(title="rho-tts Studio", theme=gr.themes.Soft()) as app:

        # ── shared helpers ──────────────────────────────────────────

        def _voice_choices(model_id=None):
            return gr.Dropdown(
                choices=callbacks.voice_choices_for_model(state.config, model_id),
            )

        def _model_choices():
            return gr.Dropdown(choices=callbacks.model_choices(state.config))

        def _refresh_dropdowns(model_id=None):
            return _model_choices(), _voice_choices(model_id)

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
                    _model_list = callbacks.model_choices(state.config)
                    _initial_model_id = _model_list[0][1] if _model_list else None
                    model_dd = gr.Dropdown(
                        choices=_model_list,
                        value=_initial_model_id,
                        label="Model",
                        interactive=True,
                    )
                    _initial_voice_choices = callbacks.voice_choices_for_model(state.config, _initial_model_id)
                    voice_dd = gr.Dropdown(
                        choices=_initial_voice_choices,
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
                    _initial_status = callbacks.status_for_model_voice(
                        state.config, _initial_model_id, _initial_voice_choices,
                    )
                    status_box = gr.Textbox(label="Status", interactive=False, value=_initial_status)
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

            def _on_voice_change(voice_id, model_id):
                """Reload phonetic mapping when voice changes."""
                rows, label = callbacks.load_phonetic_mapping(state, voice_id, model_id)
                return rows, label

            def _on_model_change(voice_id, model_id):
                """Filter voices for provider and reload phonetic mapping."""
                choices = callbacks.voice_choices_for_model(state.config, model_id)
                valid_ids = {c[1] for c in choices}
                new_voice_id = voice_id if voice_id in valid_ids else None
                rows, label = callbacks.load_phonetic_mapping(
                    state, new_voice_id, model_id,
                )
                status = callbacks.status_for_model_voice(state.config, model_id, choices)
                return (
                    gr.Dropdown(choices=choices, value=new_voice_id),
                    rows,
                    label,
                    status,
                )

            voice_dd.change(
                fn=_on_voice_change,
                inputs=[voice_dd, model_dd],
                outputs=[mapping_df, mapping_label],
            )
            model_dd.change(
                fn=_on_model_change,
                inputs=[voice_dd, model_dd],
                outputs=[voice_dd, mapping_df, mapping_label, status_box],
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

            def _add_voice(vid, vname, vaudio, vref, current_model_id):
                table, msg = callbacks.add_voice(state, vid, vname, vaudio, vref)
                new_voice_dd = _voice_choices(current_model_id)
                status = callbacks.status_for_model_voice(
                    state.config, current_model_id, new_voice_dd.choices,
                )
                return table, msg, new_voice_dd, _model_choices(), status

            v_add_btn.click(
                fn=_add_voice,
                inputs=[v_id, v_name, v_audio, v_ref_text, model_dd],
                outputs=[v_table, v_status, voice_dd, model_dd, status_box],
            )

            def _del_voice(vid, current_model_id):
                table, msg = callbacks.delete_voice(state, vid)
                new_voice_dd = _voice_choices(current_model_id)
                status = callbacks.status_for_model_voice(
                    state.config, current_model_id, new_voice_dd.choices,
                )
                return table, msg, new_voice_dd, _model_choices(), status

            v_del_btn.click(
                fn=_del_voice,
                inputs=[v_del_id, model_dd],
                outputs=[v_table, v_status, voice_dd, model_dd, status_box],
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
            m_name = gr.Textbox(
                label="Model Name",
                placeholder="(uses catalog name)",
                info="Optional friendly name for this model configuration",
            )
            with gr.Row():
                m_download_status = gr.Textbox(
                    label="Download Status", interactive=False, visible=False,
                )
                m_download_btn = gr.Button(
                    "Download Model", variant="secondary", visible=False,
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
                return (
                    gr.Dropdown(choices=choices, value=None, interactive=True),
                    gr.Textbox(visible=False),
                    gr.Button(visible=False),
                    gr.Textbox(placeholder="(uses catalog name)", value=""),
                )

            m_provider.change(
                fn=_on_provider_change,
                inputs=[m_provider],
                outputs=[m_model_select, m_download_status, m_download_btn, m_name],
            )

            def _on_model_select_change(provider, model_name):
                """Fill threshold defaults and check download status when a catalog model is selected."""
                if not provider or not model_name:
                    return (
                        gr.skip(), gr.skip(), gr.skip(),
                        gr.Textbox(visible=False), gr.Button(visible=False),
                        gr.Textbox(placeholder="(uses catalog name)"),
                    )
                defaults = get_provider_model_defaults(provider, model_name)
                model_path = defaults.get("model_path")

                # Download status
                if model_path:
                    cached = is_model_cached(model_path)
                    dl_status = gr.Textbox(
                        value="Downloaded" if cached else "Not downloaded",
                        visible=True,
                    )
                    dl_btn = gr.Button(visible=not cached)
                else:
                    dl_status = gr.Textbox(visible=False)
                    dl_btn = gr.Button(visible=False)

                return (
                    defaults.get("max_iterations", 10),
                    defaults.get("accent_drift_threshold", 0.17),
                    defaults.get("text_similarity_threshold", 0.85),
                    dl_status,
                    dl_btn,
                    gr.Textbox(placeholder=f"(default: {model_name})"),
                )

            m_model_select.change(
                fn=_on_model_select_change,
                inputs=[m_provider, m_model_select],
                outputs=[m_max_iter, m_accent, m_text_sim, m_download_status, m_download_btn, m_name],
            )

            def _download_model(provider, model_name):
                return callbacks.download_model(provider, model_name)

            m_download_btn.click(
                fn=_download_model,
                inputs=[m_provider, m_model_select],
                outputs=[m_download_status, m_download_btn],
            )

            # -- Model events --

            def _add_model(mprov, mmodel, miter, macc, mtsim, mname, current_model_id):
                table, msg = callbacks.add_model(
                    state, mprov, mmodel, miter, macc, mtsim,
                    custom_name=mname,
                )
                return (
                    table,
                    msg,
                    _model_choices(),
                    _voice_choices(current_model_id),
                    gr.Dropdown(choices=callbacks.model_choices(state.config)),
                )

            m_add_btn.click(
                fn=_add_model,
                inputs=[m_provider, m_model_select, m_max_iter, m_accent, m_text_sim, m_name, model_dd],
                outputs=[m_table, m_status, model_dd, voice_dd, m_del_dd],
            )

            def _del_model(model_id, current_model_id):
                table, msg = callbacks.delete_model(state, model_id)
                return (
                    table,
                    msg,
                    _model_choices(),
                    _voice_choices(current_model_id),
                    gr.Dropdown(choices=callbacks.model_choices(state.config)),
                )

            m_del_btn.click(
                fn=_del_model,
                inputs=[m_del_dd, model_dd],
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
