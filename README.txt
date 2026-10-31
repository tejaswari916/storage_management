SMART STORAGE MANAGEMENT FOR PHOTOS & VIDEOS

HOW TO RUN ON MAC:
1. Extract this ZIP file.
2. Open Terminal and cd into the extracted smart_photo_storage_app folder.
3. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate
4. Install dependencies:
   pip install -r requirements.txt
5. Run:
   streamlit run app.py
6. Browser usually opens at http://localhost:8501

HOW TO USE:
- Choose "Upload photos/videos" and select files from extracted folders, OR
- Choose "Upload a ZIP folder" and upload the generated dataset ZIP.
- Click "Analyze and add files".
- Review the latest analysis and dashboard.

IMPORTANT LIMITATIONS:
- Exact duplicates are detected using SHA-256.
- Similar images are suggested using perceptual hashing; false positives are possible.
- Blur detection is a simple heuristic and can mislabel some images.
- Screenshot detection is heuristic-based and is not a reliable general-purpose AI classifier.
- "Large file" is based on the threshold in the sidebar.
- Video content is not analyzed; videos are categorized by file type and size.
- This is a local prototype. Files remain in stored_files and are not sent to a cloud service.
- No file is deleted automatically.
