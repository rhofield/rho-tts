"""
Gradio Blocks application for the rho-tts web UI.

Five-tab layout:
  1. Generate  — model/voice selection, text input, audio playback, phonetic mappings
  2. Voices   — manage voice profiles (upload reference audio, set transcript)
  3. Models   — manage model configurations (provider params, thresholds)
  4. Training — train the accent drift classifier on labeled audio
  5. Library  — browse, filter, and replay past generations
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

    with gr.Blocks(title="rho-tts Studio") as app:

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

        with gr.Tab("Generate") as generate_tab:
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
                    column_count=(2, "fixed"),
                    interactive=True,
                    row_count=(3, "dynamic"),
                    label="Phonetic Mappings",
                )
                with gr.Row():
                    save_mapping_btn = gr.Button("Save Mappings")
                    mapping_status = gr.Textbox(label="", interactive=False, show_label=False)

            # -- Parameters accordion --
            with gr.Accordion("Parameters", open=False):
                params_label = gr.Markdown("Select a voice and model to load parameters.")
                with gr.Row():
                    params_seed = gr.Number(label="seed", value=789, precision=0)
                    params_max_iter = gr.Number(label="max_iterations", value=10, precision=0)
                with gr.Row():
                    params_accent = gr.Slider(
                        label="accent_drift_threshold", minimum=0, maximum=1, step=0.01, value=0.17,
                    )
                    params_text_sim = gr.Slider(
                        label="text_similarity_threshold", minimum=0, maximum=1, step=0.01, value=0.85,
                    )
                with gr.Row():
                    params_temperature = gr.Slider(
                        label="temperature", minimum=0.1, maximum=2.0, step=0.05, value=1.0, visible=False,
                    )
                    params_cfg_weight = gr.Slider(
                        label="cfg_weight", minimum=0.1, maximum=1.0, step=0.05, value=0.6, visible=False,
                    )
                with gr.Row():
                    save_params_btn = gr.Button("Save Parameters")
                    params_status = gr.Textbox(label="", interactive=False, show_label=False)

            # -- Generate tab events --

            def _on_device_change(device):
                state.config.device = device
                state.save()
                state.invalidate_tts()

            device_dd.change(fn=_on_device_change, inputs=[device_dd])

            def _on_voice_change(voice_id, model_id):
                """Reload phonetic mapping and params when voice changes."""
                rows, mapping_label_val = callbacks.load_phonetic_mapping(state, voice_id, model_id)
                seed, max_iter, accent, text_sim, temp, cfg, plabel, is_cb = \
                    callbacks.load_model_voice_params(state, voice_id, model_id)
                return (
                    rows, mapping_label_val,
                    seed, max_iter, accent, text_sim,
                    gr.update(value=temp, visible=is_cb),
                    gr.update(value=cfg, visible=is_cb),
                    plabel,
                )

            def _on_model_change(voice_id, model_id):
                """Filter voices for provider and reload phonetic mapping and params."""
                choices = callbacks.voice_choices_for_model(state.config, model_id)
                valid_ids = {c[1] for c in choices}
                new_voice_id = voice_id if voice_id in valid_ids else None
                rows, mapping_label_val = callbacks.load_phonetic_mapping(
                    state, new_voice_id, model_id,
                )
                seed, max_iter, accent, text_sim, temp, cfg, plabel, is_cb = \
                    callbacks.load_model_voice_params(state, new_voice_id, model_id)
                status = callbacks.status_for_model_voice(state.config, model_id, choices)
                return (
                    gr.Dropdown(choices=choices, value=new_voice_id),
                    rows,
                    mapping_label_val,
                    status,
                    seed, max_iter, accent, text_sim,
                    gr.update(value=temp, visible=is_cb),
                    gr.update(value=cfg, visible=is_cb),
                    plabel,
                )

            voice_dd.change(
                fn=_on_voice_change,
                inputs=[voice_dd, model_dd],
                outputs=[
                    mapping_df, mapping_label,
                    params_seed, params_max_iter, params_accent, params_text_sim,
                    params_temperature, params_cfg_weight, params_label,
                ],
            )
            model_dd.change(
                fn=_on_model_change,
                inputs=[voice_dd, model_dd],
                outputs=[
                    voice_dd, mapping_df, mapping_label, status_box,
                    params_seed, params_max_iter, params_accent, params_text_sim,
                    params_temperature, params_cfg_weight, params_label,
                ],
            )

            def _save_params(voice_id, model_id, seed, max_iter, accent, text_sim, temp, cfg):
                return callbacks.save_model_voice_params(
                    state, voice_id, model_id, seed, max_iter, accent, text_sim, temp, cfg,
                )

            save_params_btn.click(
                fn=_save_params,
                inputs=[
                    voice_dd, model_dd,
                    params_seed, params_max_iter, params_accent, params_text_sim,
                    params_temperature, params_cfg_weight,
                ],
                outputs=[params_status],
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

        with gr.Tab("Voices") as voices_tab:
            gr.Markdown("### Manage Voice Profiles")
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
                v_del_dd = gr.Dropdown(
                    choices=callbacks.user_voice_choices(state.config),
                    label="Voice to Delete",
                    interactive=True,
                    scale=2,
                )
                v_del_btn = gr.Button("Delete Voice", variant="stop", scale=0, min_width=140)

            # -- Voice events --

            def _add_voice(vname, vaudio, vref, current_model_id):
                table, msg = callbacks.add_voice(state, vname, vaudio, vref)
                new_voice_dd = _voice_choices(current_model_id)
                status = callbacks.status_for_model_voice(
                    state.config, current_model_id, new_voice_dd.choices,
                )
                del_choices = callbacks.user_voice_choices(state.config)
                return table, msg, new_voice_dd, _model_choices(), status, gr.Dropdown(choices=del_choices)

            v_add_btn.click(
                fn=_add_voice,
                inputs=[v_name, v_audio, v_ref_text, model_dd],
                outputs=[v_table, v_status, voice_dd, model_dd, status_box, v_del_dd],
            )

            def _del_voice(vid, current_model_id):
                table, msg = callbacks.delete_voice(state, vid)
                new_voice_dd = _voice_choices(current_model_id)
                status = callbacks.status_for_model_voice(
                    state.config, current_model_id, new_voice_dd.choices,
                )
                del_choices = callbacks.user_voice_choices(state.config)
                return table, msg, new_voice_dd, _model_choices(), status, gr.Dropdown(choices=del_choices)

            v_del_btn.click(
                fn=_del_voice,
                inputs=[v_del_dd, model_dd],
                outputs=[v_table, v_status, voice_dd, model_dd, status_box, v_del_dd],
            )

        # ── Tab 3: Models ───────────────────────────────────────────

        with gr.Tab("Models") as models_tab:
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
                with gr.Accordion("Quality Control", open=True):
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

            with gr.Accordion("Edit Model", open=False):
                m_edit_dd = gr.Dropdown(
                    choices=callbacks.model_choices(state.config),
                    label="Model to Edit",
                    interactive=True,
                )
                m_edit_name = gr.Textbox(label="Model Name", interactive=True)
                with gr.Accordion("Quality Control", open=True):
                    with gr.Row():
                        m_edit_iter = gr.Number(label="max_iterations", value=10, precision=0)
                        m_edit_accent = gr.Slider(
                            label="accent_drift_threshold",
                            minimum=0, maximum=1, step=0.01, value=0.17,
                        )
                        m_edit_sim = gr.Slider(
                            label="text_similarity_threshold",
                            minimum=0, maximum=1, step=0.01, value=0.85,
                        )
                m_edit_btn = gr.Button("Save Changes", variant="primary")
                m_edit_status = gr.Textbox(label="Status", interactive=False)

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

            def _refresh_model_dropdowns(edit_value=None):
                """Return updated choices for delete and edit dropdowns."""
                choices = callbacks.model_choices(state.config)
                return (
                    gr.Dropdown(choices=choices),
                    gr.Dropdown(choices=choices, value=edit_value),
                )

            def _add_model(mprov, mmodel, miter, macc, mtsim, mname, current_model_id):
                table, msg = callbacks.add_model(
                    state, mprov, mmodel, miter, macc, mtsim,
                    custom_name=mname,
                )
                del_dd, edit_dd = _refresh_model_dropdowns()
                return (
                    table,
                    msg,
                    _model_choices(),
                    _voice_choices(current_model_id),
                    del_dd,
                    edit_dd,
                )

            m_add_btn.click(
                fn=_add_model,
                inputs=[m_provider, m_model_select, m_max_iter, m_accent, m_text_sim, m_name, model_dd],
                outputs=[m_table, m_status, model_dd, voice_dd, m_del_dd, m_edit_dd],
            )

            # -- Edit model events --

            def _on_edit_model_select(model_id):
                name, iters, accent, sim = callbacks.load_model_for_edit(state, model_id)
                return name, iters, accent, sim

            m_edit_dd.change(
                fn=_on_edit_model_select,
                inputs=[m_edit_dd],
                outputs=[m_edit_name, m_edit_iter, m_edit_accent, m_edit_sim],
            )

            def _edit_model(model_id, name, iters, accent, sim, current_model_id):
                table_data, msg = callbacks.edit_model(
                    state, model_id, name, iters, accent, sim,
                )
                updated_name, updated_iters, updated_accent, updated_sim = (
                    callbacks.load_model_for_edit(state, model_id)
                )
                choices = callbacks.model_choices(state.config)
                return (
                    gr.Dataframe(value=table_data),
                    msg,
                    _model_choices(),
                    _voice_choices(current_model_id),
                    gr.Dropdown(choices=choices),
                    gr.Dropdown(choices=choices, value=model_id),
                    updated_name,
                    updated_iters,
                    updated_accent,
                    updated_sim,
                )

            m_edit_btn.click(
                fn=_edit_model,
                inputs=[m_edit_dd, m_edit_name, m_edit_iter, m_edit_accent, m_edit_sim, model_dd],
                outputs=[
                    m_table, m_edit_status, model_dd, voice_dd,
                    m_del_dd, m_edit_dd,
                    m_edit_name, m_edit_iter, m_edit_accent, m_edit_sim,
                ],
            )

            # -- Delete model events --

            def _del_model(model_id, current_model_id):
                table, msg = callbacks.delete_model(state, model_id)
                del_dd, edit_dd = _refresh_model_dropdowns()
                return (
                    table,
                    msg,
                    _model_choices(),
                    _voice_choices(current_model_id),
                    del_dd,
                    edit_dd,
                )

            m_del_btn.click(
                fn=_del_model,
                inputs=[m_del_dd, model_dd],
                outputs=[m_table, m_status, model_dd, voice_dd, m_del_dd, m_edit_dd],
            )

        # ── Tab 4: Training ─────────────────────────────────────────

        with gr.Tab("Training") as training_tab:
            gr.Markdown("## Accent Drift Classifier Training")
            gr.Markdown(
                "Train the classifier on labeled `.wav` samples.\n\n"
                "Dataset must contain two subdirectories:\n"
                "- `good/` — samples **without** accent drift\n"
                "- `bad/`  — samples **with** accent drift"
            )
            _train_voice_choices = callbacks.voice_choices(state.config)
            t_voice_dd = gr.Dropdown(
                choices=_train_voice_choices,
                label="Voice",
                interactive=True,
            )
            t_dataset_dir = gr.Textbox(
                label="Dataset Directory",
                placeholder="/path/to/dataset",
            )
            t_train_btn = gr.Button("Train Classifier", variant="primary")
            t_log = gr.Textbox(
                label="Training Log",
                lines=12,
                interactive=False,
            )
            t_status = gr.Textbox(label="", interactive=False, show_label=False)

            def _do_train(voice_id, ds):
                yield from callbacks.train_classifier(state, voice_id, ds)

            t_train_btn.click(
                fn=_do_train,
                inputs=[t_voice_dd, t_dataset_dir],
                outputs=[t_log, t_status],
            )

        def _on_training_tab():
            return gr.Dropdown(choices=callbacks.voice_choices(state.config))

        training_tab.select(fn=_on_training_tab, outputs=[t_voice_dd])

        # ── Tab 5: Library ──────────────────────────────────────────

        with gr.Tab("Library") as library_tab:
            gr.Markdown("### Audio Library")
            with gr.Row():
                lib_model_dd = gr.Dropdown(
                    choices=callbacks.library_model_choices(state),
                    label="Filter by Model",
                    interactive=True,
                    scale=1,
                )
                lib_voice_dd = gr.Dropdown(
                    choices=callbacks.library_voice_choices(state),
                    label="Filter by Voice",
                    interactive=True,
                    scale=1,
                )
                lib_text_search = gr.Textbox(
                    label="Search Text",
                    placeholder="Substring search...",
                    scale=1,
                )
                lib_filter_btn = gr.Button("Apply Filters", scale=0, min_width=130)

            lib_table = gr.Dataframe(
                value=callbacks.library_table(state),
                headers=["Timestamp", "Model", "Voice", "Text", "Duration", "Status", "ID"],
                interactive=False,
                label="Generation History",
                column_widths=["150px", "120px", "120px", "1fr", "70px", "90px", "0px"],
            )

            with gr.Row():
                lib_audio = gr.Audio(label="Playback", type="filepath", interactive=False)
                lib_transcript = gr.Textbox(
                    label="Full Transcript",
                    interactive=False,
                    lines=4,
                )

            with gr.Row():
                lib_del_btn = gr.Button("Delete Selected", variant="stop", scale=0, min_width=150)
                lib_clear_btn = gr.Button("Clear All History", variant="stop", scale=0, min_width=160)
                lib_status = gr.Textbox(label="Status", interactive=False, scale=2)

            # Hidden state to track the currently selected record ID
            lib_selected_id = gr.State(value=None)

            # -- Library events --

            def _lib_apply_filters(model_f, voice_f, text_s):
                return callbacks.library_table(state, model_f, voice_f, text_s)

            lib_filter_btn.click(
                fn=_lib_apply_filters,
                inputs=[lib_model_dd, lib_voice_dd, lib_text_search],
                outputs=[lib_table],
            )

            def _lib_on_select(table_data, evt: gr.SelectData):
                """When a row is clicked, load its audio and transcript."""
                row_idx = evt.index[0]
                if hasattr(table_data, "values"):
                    rows = table_data.values.tolist()
                else:
                    rows = table_data
                if row_idx < 0 or row_idx >= len(rows):
                    return None, "", None
                record_id = rows[row_idx][-1]  # ID is last column
                audio_path, full_text = callbacks.library_get_audio(state, record_id)
                return audio_path, full_text, record_id

            lib_table.select(
                fn=_lib_on_select,
                inputs=[lib_table],
                outputs=[lib_audio, lib_transcript, lib_selected_id],
            )

            def _lib_delete(selected_id, model_f, voice_f, text_s):
                if not selected_id:
                    return (
                        callbacks.library_table(state, model_f, voice_f, text_s),
                        "No record selected.",
                        None, None, "",
                        gr.Dropdown(choices=callbacks.library_model_choices(state)),
                        gr.Dropdown(choices=callbacks.library_voice_choices(state)),
                    )
                msg = callbacks.library_delete_record(state, selected_id)
                return (
                    callbacks.library_table(state, model_f, voice_f, text_s),
                    msg,
                    None, None, "",
                    gr.Dropdown(choices=callbacks.library_model_choices(state)),
                    gr.Dropdown(choices=callbacks.library_voice_choices(state)),
                )

            lib_del_btn.click(
                fn=_lib_delete,
                inputs=[lib_selected_id, lib_model_dd, lib_voice_dd, lib_text_search],
                outputs=[lib_table, lib_status, lib_selected_id, lib_audio, lib_transcript, lib_model_dd, lib_voice_dd],
            )

            def _lib_clear(model_f, voice_f, text_s):
                msg = callbacks.library_clear_history(state)
                return (
                    callbacks.library_table(state, model_f, voice_f, text_s),
                    msg,
                    None, None, "",
                    gr.Dropdown(choices=[]),
                    gr.Dropdown(choices=[]),
                )

            lib_clear_btn.click(
                fn=_lib_clear,
                inputs=[lib_model_dd, lib_voice_dd, lib_text_search],
                outputs=[lib_table, lib_status, lib_selected_id, lib_audio, lib_transcript, lib_model_dd, lib_voice_dd],
            )

        # ── Tab select: hydrate components from current config ────

        def _on_generate_tab():
            models = callbacks.model_choices(state.config)
            initial_model_id = models[0][1] if models else None
            voices = callbacks.voice_choices_for_model(state.config, initial_model_id)
            status = callbacks.status_for_model_voice(
                state.config, initial_model_id, voices,
            )
            return (
                gr.Dropdown(choices=models, value=initial_model_id),
                gr.Dropdown(choices=voices),
                status,
            )

        generate_tab.select(
            fn=_on_generate_tab,
            outputs=[model_dd, voice_dd, status_box],
        )

        def _on_voices_tab():
            return (
                callbacks._voices_table(state.config),
                gr.Dropdown(choices=callbacks.user_voice_choices(state.config)),
            )

        voices_tab.select(fn=_on_voices_tab, outputs=[v_table, v_del_dd])

        def _on_models_tab():
            choices = callbacks.model_choices(state.config)
            return (
                callbacks._models_table(state.config),
                gr.Dropdown(choices=choices),
                gr.Dropdown(choices=choices),
            )

        models_tab.select(
            fn=_on_models_tab,
            outputs=[m_table, m_del_dd, m_edit_dd],
        )

        def _on_library_tab():
            return (
                callbacks.library_table(state),
                gr.Dropdown(choices=callbacks.library_model_choices(state)),
                gr.Dropdown(choices=callbacks.library_voice_choices(state)),
            )

        library_tab.select(
            fn=_on_library_tab,
            outputs=[lib_table, lib_model_dd, lib_voice_dd],
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
    app.launch(server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft())
