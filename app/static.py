import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

import gradio as gr
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

API_BASE_URL = "http://127.0.0.1:8000"

def fetch_all_sessions():
    try:
        res = requests.get(f"{API_BASE_URL}/sessions", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if not data:
                return [["-", "Belum ada sesi tersimpan", "-"]]
            
            table_data = []
            for s in data:
                table_data.append([s['id'], s['food_label'].upper(), s['start_date']])
            return table_data
        return [["-", "Gagal mengambil data sesi", "-"]]
    except:
        return [["-", "Server API belum nyala (Uvicorn mati)", "-"]]

def handle_predict(image_path, session_id):
    if not image_path:
        return "❌ Upload foto dulu!", gr.update(value=[], visible=False), gr.update(value=None), session_id, gr.update()

    data = {"session_id": session_id.strip() if session_id else "", "storage_type": "room_temp"}
    
    with open(image_path, "rb") as img_file:
        files = {"file": ("img.jpg", img_file, "image/jpeg")}
        try:
            res = requests.post(f"{API_BASE_URL}/predict", data=data, files=files, timeout=60)
            result = res.json()
            
            if res.status_code != 200:
                return f"❌ Error Server: {result.get('detail')}", gr.update(value=[], visible=False), gr.update(value=None), session_id, fetch_all_sessions()

            curr_id = result.get('session_id')
            status_freshness = result.get('freshness_status')
            
            # 1. Ambil label AI (jika ada) untuk pesan penolakan
            if curr_id == "rejected" or status_freshness == "unknown":
                lbl = result.get('food_label')
                print(lbl)
                label_asing = lbl.upper() if lbl else "Maaf benda belum masuk ke dataset"
                pesan_tolak = f"Rejected: {label_asing}, sistem belum mengenali makanan ini"
                return pesan_tolak, gr.update(value=[], visible=False), gr.update(value=None), session_id, fetch_all_sessions()

            # 2. Cek mismatch session_id (jika ada) untuk pesan penolakan
            if status_freshness == "error_mismatch":
                pesan_mismatch = result.get('recommendation', 'Label tidak cocok dengan ID sesi ini.')
                return pesan_mismatch, gr.update(value=[], visible=False), gr.update(value=None), session_id, fetch_all_sessions()

            # 3. Ambil riwayat sesi untuk tampilan gallery & plot
            hist_res = requests.get(f"{API_BASE_URL}/session/{curr_id}/history", timeout=5)
            
            if hist_res.status_code != 200:
                return "✅ Foto tersimpan", gr.update(), gr.update(), curr_id, fetch_all_sessions()
                
            hist_data = hist_res.json()
            
            gallery_imgs = []
            for item in hist_data.get('history', []):
                try:
                    img_res = requests.get(item['image_url'])
                    if img_res.status_code == 200:
                        gallery_imgs.append((Image.open(BytesIO(img_res.content)), f"Hari {item['day']}"))
                except Exception as e:
                    continue

            status_txt = f"✅ SESSION ID: {curr_id}\n"
            status_txt += f"MAKANAN: {hist_data.get('food_label', '').upper()}\n"
            status_txt += f"PROGRESS: {hist_data.get('total_days', 0)}/3 Foto\n"

            # 4. Tampilkan status prediksi dan plot degradasi jika data cukup, atau pesan pengumpulan data jika belum cukup
            fig = None
            if status_freshness != "collecting_data":
                sisa = result.get('days_remaining', 0.0)
                
                is_basi = status_freshness.lower() in ["spoiled", "basi", "rotten"]
                
                if is_basi:
                    status_txt += f"\n\nPERINGATAN: MAKANAN BASI!\n🎯 PREDIKSI: {sisa:.1f} Hari Lagi ({status_freshness.upper()})"
                    line_color = '#5D4037'
                else:
                    status_txt += f"\n\nSTATUS AMAN\n🎯 PREDIKSI: {sisa:.1f} Hari Lagi ({status_freshness.upper()})"
                    line_color = '#10B981'
                
                # Plot curve
                fig, ax = plt.subplots(figsize=(5, 3))
                x = list(range(1, hist_data['total_days'] + 1))
                y = [sisa + (len(x)-i)*1.2 for i in x] 
                ax.plot(x, y, 'o-', color=line_color, linewidth=2.5) 
                ax.set_title(f"Estimasi Degradasi {hist_data.get('food_label', '').upper()}")
                ax.set_xlabel("Hari Pengamatan")
                ax.set_ylabel("Sisa Hari")
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
            else:
                status_txt += "\n⏳ Sistem masih mengumpulkan data (Butuh min 3 hari beruntun)"

            return status_txt, gr.update(value=gallery_imgs, visible=True), gr.update(value=fig, visible=True if fig else False), curr_id, fetch_all_sessions()

        except Exception as e:
            return f"❌ Koneksi Gagal: {e}", gr.update(value=[], visible=False), gr.update(value=None), session_id, gr.update()

def reset_ui():
    return None, "", "Form direset. Silakan mulai sesi baru.", gr.update(value=[], visible=False), gr.update(value=None)


theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="green",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#F8FAFC", 
    block_background_fill="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E2E8F0",
    button_primary_background_fill="#10B981",
    button_primary_background_fill_hover="#059669",
    button_primary_text_color="#FFFFFF",
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# Food Decay Prediction 🍎🕒")
    gr.Markdown("Description in english : This app predicts the shelf life of food items based on daily photos. Upload a photo of your food, and the AI will analyze its freshness over time, providing insights on how long it will remain safe to consume. Perfect for reducing food waste and making informed decisions about your groceries!")
    
    with gr.Tab("📸 Upload & Predict"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="filepath", label="Ambil Foto Buah")
                sess_in = gr.Textbox(label="Session ID Aktif", placeholder="Akan terisi otomatis atau masukkan ID lama")
                with gr.Row():
                    btn = gr.Button("Proses Foto", variant="primary")
                    clear_btn = gr.Button("🗑️ Reset Form", variant="secondary") 
            
            with gr.Column():
                status_out = gr.Textbox(label="Data Makanan", lines=7)
                gallery_out = gr.Gallery(label="Timeline Foto", columns=3, visible=False)
                plot_out = gr.Plot(label="Analisis Degradasi", visible=False)

    with gr.Tab("🗄️ Database Session Manager"):
        refresh_btn = gr.Button("🔄 Refresh Daftar Sesi")
        
        session_list_out = gr.Dataframe(
            headers=["Session ID", "Label Makanan", "Tanggal Mulai"],
            datatype=["str", "str", "str"],
            label="Semua Sesi Tersimpan",
            interactive=False, 
            value=fetch_all_sessions()
        )
    
    # Trigger predict
    btn.click(
        fn=handle_predict, 
        inputs=[img_in, sess_in], 
        outputs=[status_out, gallery_out, plot_out, sess_in, session_list_out]
    )
    
    # Trigger refresh database
    refresh_btn.click(fn=fetch_all_sessions, inputs=None, outputs=session_list_out)
    
    # Trigger reset form
    clear_btn.click(
        fn=reset_ui, 
        inputs=None, 
        outputs=[img_in, sess_in, status_out, gallery_out, plot_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)