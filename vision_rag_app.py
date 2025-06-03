import streamlit as st
import cohere
import google.genai as genai
import requests
import os
import io
import base64
import PIL
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import chromadb
from PIL import Image
import time

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("CO_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize APIs
co = cohere.ClientV2(api_key=cohere_api_key)
client = genai.Client(api_key=gemini_api_key)

# Configuration
IMAGE_DIR = "img"
os.makedirs(IMAGE_DIR, exist_ok=True)
MAX_PIXELS = 1568 * 1568  # Max resolution for images
CHROMA_DB_PATH = "chroma_db"  # Directory for ChromaDB persistence

# Initialize ChromaDB with persistence
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="vision_docs",
    embedding_function=None  # We'll handle embeddings ourselves
)

# Sample images from AppEconomyInsights
IMAGES = {
    "tesla.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fbef936e6-3efa-43b3-88d7-7ec620cdb33b_2744x1539.png",
    "netflix.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F23bd84c9-5b62-4526-b467-3088e27e4193_2744x1539.png",
    "nike.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa5cd33ba-ae1a-42a8-a254-d85e690d9870_2741x1541.png",
    "google.png": "https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F395dd3b9-b38e-4d1f-91bc-d37b642ee920_2741x1541.png",
    "accenture.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F08b2227c-7dc8-49f7-b3c5-13cab5443ba6_2741x1541.png",
    "tecent.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0ec8448c-c4d1-4aab-a8e9-2ddebe0c95fd_2741x1541.png",
    "alibaba.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc45d3209-50cc-4255-846c-4bc20ee90129_2745x1539.png",
    "blackrock.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F05e6571e-93c6-4f13-8253-e31d688668f5_2459x1377.png",
    "disney.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8db10610-1d8b-420a-8277-50336c08b0db_2745x1539.png",
    "walmart.png": "https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1c4e6c9a-dc50-4b59-903f-97be4c67517e_2745x1539.png",
    "actors.png": "http://blogtobollywood.com/wp-content/uploads/2014/06/highestpaidactors-553x1024.jpg"
}

# Helper functions
def resize_image(pil_image):
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path):
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"
    resize_image(pil_image)
    
    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64," + base64.b64encode(img_buffer.read()).decode("utf-8")
    return img_data

def embed_image(img_path):
    """Generate embedding for an image using Cohere v4"""
    img_data = base64_from_image(img_path)
    response = co.embed(
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
        inputs=[{"content": [{"type": "image", "image": img_data}]}]
    )
    return response.embeddings.float[0]

def embed_text(text):
    """Generate embedding for text using Cohere v4"""
    response = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[text]
    )
    return response.embeddings.float[0]

def initialize_database(images_dict=None):
    """Initialize database only with provided images"""
    img_paths = []
    existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
    
    # Only process if images are provided
    if images_dict:
        for name, url in tqdm(images_dict.items()):
            img_path = os.path.join(IMAGE_DIR, name)
            img_paths.append(img_path)
            
            if not os.path.exists(img_path):
                response = requests.get(url)
                response.raise_for_status()
                with open(img_path, "wb") as f:
                    f.write(response.content)
            
            if name not in existing_ids:
                embedding = embed_image(img_path)
                collection.add(
                    ids=[name],
                    embeddings=[embedding],
                    metadatas=[{"path": img_path}],
                    documents=[name]
                )
    
    return img_paths

# Initialize database on first run
if 'img_paths' not in st.session_state:
    st.session_state.img_paths = []

# RAG functions
def search(question, max_img_size=800):
    # First embed the question
    query_embedding = embed_text(question)
    
    # Query ChromaDB with the embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["metadatas"]
    )
    
    hit_img_path = results['metadatas'][0][0]['path']
    image = PIL.Image.open(hit_img_path)
    image.thumbnail((max_img_size, max_img_size))
    return hit_img_path, image

def answer(question, img_path):
    timing_metrics = {}
    start_time = time.time()
    image = PIL.Image.open(img_path)
    timing_metrics["Image Prep"] = time.time() - start_time
    prompt = [f"""Answer the question based on the following image.
              # Don't use markdown.
              # Please provide enough context for your answer. {question}""", image]
    start_time = time.time()
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        #model="gemini-1.5-flash",
        contents=prompt
    )
    timing_metrics["Total Gemini Time"] = time.time() - start_time
    try:
        if hasattr(response, '_raw_response'):
            timing_metadata = response._raw_response.metadata  # Google's internal timing
            if 'token_latency' in timing_metadata:
                timing_metrics["Token Generation"] = sum(t['latency'] for t in timing_metadata['token_latency'])
            if 'first_token_latency' in timing_metadata:
                timing_metrics["First Token"] = timing_metadata['first_token_latency']
    except Exception as e:
        st.warning(f"Couldn't extract detailed timing: {str(e)}")
    
    return response.text, timing_metrics

def handle_local_upload(uploaded_file, image_key):
    """Process and save locally uploaded image with key"""
    try:
        # Create image directory if not exists
        os.makedirs(IMAGE_DIR, exist_ok=True)
        
        # Determine file extension
        file_ext = uploaded_file.name.split('.')[-1].lower()
        valid_extensions = ['png', 'jpg', 'jpeg']
        
        if file_ext not in valid_extensions:
            st.error("Invalid file type. Please upload PNG, JPG, or JPEG.")
            return None
        
        # Save the image with the provided key
        img_path = os.path.join(IMAGE_DIR, f"{image_key}.{file_ext}")
        
        # Convert to PIL Image and resize if needed
        pil_image = Image.open(uploaded_file)
        resize_image(pil_image)
        
        # Save the processed image
        pil_image.save(img_path)
        
        return img_path
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Vision-RAG System", layout="wide")
st.title("📊 Vision-RAG with Cohere Embed v4 + Gemini Flash")

# Initialize session state for query
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Query Images", 
    "ℹ️ Sample Queries", 
    "📤 Upload Local Images",
    "📁 Folder Upload",
    "🗑️ Delete Database"
])

with tab1:
    st.header("Ask Questions About Infographics")
    user_query = st.text_input("Enter your question", value=st.session_state.current_query, key="user_query")
    
    if st.button("Search and Answer"):
        if user_query:
            total_time = 0
            with st.spinner("Searching for relevant image..."):
                try:
                    start_time = time.time()
                    img_path, image = search(user_query)
                    search_time = time.time() - start_time
                    total_time += search_time
                    
                    st.image(image, caption="Most Relevant Image")
                    
                    with st.spinner("Generating answer..."):
                        start_time = time.time()
                        answer_text, gemini_timings = answer(user_query, img_path)
                        answer_time = time.time() - start_time
                        total_time += answer_time
                    
                    st.markdown("### Answer")
                    st.write(answer_text)

                    st.markdown("### Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Image Search", f"{search_time:.2f}s")
                    col2.metric("Answer Generation", f"{answer_time:.2f}s")
                    col3.metric("Total Time", f"{total_time:.2f}s")
                    with st.expander("🔍 Detailed Gemini Timing"):
                        st.subheader("Gemini Response Breakdown")
                        
                        # Ensure we have the expected timing metrics
                        gemini_timings.setdefault("Image Prep", 0)
                        gemini_timings.setdefault("Total Gemini Time", answer_time)
                        
                        # Create columns for Gemini metrics
                        g_col1, g_col2, g_col3 = st.columns(3)
                        g_col1.metric("Image Preparation", f"{gemini_timings['Image Prep']:.3f}s")
                        g_col2.metric("API Roundtrip", 
                                    f"{gemini_timings['Total Gemini Time']:.3f}s",
                                    help="Network + Google processing time")
                        
                        # Show token generation metrics if available
                        if "First Token" in gemini_timings:
                            g_col3.metric("First Token Latency", 
                                        f"{gemini_timings['First Token']:.3f}s",
                                        help="Time until first word appears")
                        
                        if "Token Generation" in gemini_timings:
                            st.metric("Total Token Generation", 
                                    f"{gemini_timings['Token Generation']:.3f}s",
                                    help="Time spent generating all words")
                        
                        # Visualization
                        timing_data = {
                            "Phase": ["Image Search", "Image Prep", "Gemini Processing"],
                            "Time (s)": [search_time, 
                                        gemini_timings['Image Prep'], 
                                        gemini_timings['Total Gemini Time']]
                        }
                        st.bar_chart(timing_data, x="Phase", y="Time (s)", use_container_width=True)
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        else:
            st.warning("Please enter a question")

with tab2:
    st.header("Sample Queries")
    if st.button("Load Sample Images 📤", key="load_samples"):
        with st.spinner("Loading sample images..."):
            st.session_state.img_paths = initialize_database(IMAGES)
        st.success("Sample images loaded!")
    samples = [
        "What is the net profit for Nike?",
        "What are the 3 largest acquisitions from Google?",
        "What would be the net profit of Tesla without interest?",
        "Is GenAI a good business for consulting companies?",
        "In which region does Netflix generate the highest revenue?",
        "How much could Tencent grow their revenue year-over-year for the last 5 years?",
        "What is the net profit for Alibaba?",
        "What is the net income for Blackrock?",
        "What is segment operating income for Disney?",
        "What is the net profit for Walmart?",
        "Who is the top paid 3 bollywood actors?"
    ]
    
    for sample in samples:
        if st.button(sample, key=f"sample_{hash(sample)}"):  # Using hash for unique keys
            st.session_state.current_query = sample
            st.experimental_rerun()

with tab3:
    st.header("Upload Local Images with Key-Value Pair")
    
    # File uploader section
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=["png", "jpg", "jpeg"],
            key="file_uploader"
        )
        
    with col2:
        image_key = st.text_input(
            "Enter a unique key/name for this image",
            placeholder="e.g., company_earnings_2023",
            key="image_key_input"
        )
    
    # Display preview
    if uploaded_file is not None:
        try:
            st.subheader("Image Preview")
            preview_image = Image.open(uploaded_file)
            st.image(preview_image, caption="Upload Preview", width=300)
        except:
            st.warning("Could not display image preview")
    
    # Upload and index button
    if st.button("Upload and Index Image", key="upload_button"):
        if uploaded_file is not None and image_key:
            total_time = 0
            with st.spinner("Processing image..."):
                start_time = time.time()
                img_path = handle_local_upload(uploaded_file, image_key)
                upload_time = time.time() - start_time
                total_time += upload_time
                
                if img_path:
                    try:
                        # Generate embedding
                        start_time = time.time()
                        embedding = embed_image(img_path)
                        embed_time = time.time() - start_time
                        total_time += embed_time
                        
                        # Add to ChromaDB
                        existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
                        start_time = time.time()
                        if image_key in existing_ids:
                            collection.update(
                                ids=[image_key],
                                embeddings=[embedding],
                                metadatas=[{"path": img_path}],
                                documents=[image_key]
                            )
                        else:
                            collection.add(
                                ids=[image_key],
                                embeddings=[embedding],
                                metadatas=[{"path": img_path}],
                                documents=[image_key]
                            )
                        db_time = time.time() - start_time
                        total_time += db_time
                        
                        # Update session state
                        st.session_state.img_paths.append(img_path)
                        
                        # Display timing information
                        st.markdown("### Performance Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("File Upload", f"{upload_time:.2f}s")
                        col2.metric("Embedding", f"{embed_time:.2f}s")
                        col3.metric("DB Operation", f"{db_time:.2f}s")
                        col4.metric("Total Time", f"{total_time:.2f}s")
                        
                    except Exception as e:
                        st.error(f"Error indexing image: {str(e)}")
        else:
            st.warning("Please upload a file and provide a key/name")
    
    # Display current images in the database
    st.subheader("Current Images in Database")
    current_images = collection.get()
    
    if current_images['ids']:
        st.write(f"Total images: {len(current_images['ids'])}")
        
        # Display as a grid
        cols = st.columns(3)
        for idx, img_id in enumerate(current_images['ids']):
            with cols[idx % 3]:
                try:
                    img_path = current_images['metadatas'][idx]['path']
                    st.image(img_path, caption=img_id, use_container_width=True)
                    st.caption(f"Key: {img_id}")
                except:
                    st.warning(f"Couldn't display {img_id}")
    else:
        st.info("No images in database yet")

# ========== NEW TAB 4: FOLDER UPLOAD ==========
with tab4:
    st.header("Bulk Upload from Folder")
    
    with st.expander("📌 How to use"):
        st.write("""
        1. Create a folder with your images
        2. Make sure image filenames are meaningful (they'll be used as keys)
        3. Enter the full path to the folder below
        4. Images will be processed and added to the database
        """)
    
    folder_path = st.text_input(
        "Enter folder path (absolute path)",
        placeholder="C:/Users/name/images/",
        key="folder_path"
    )
    
    if st.button("Process Folder"):
        if folder_path and os.path.exists(folder_path):
            try:
                processed_files = 0
                skipped_files = 0
                total_time = 0
                
                with st.spinner(f"Processing images in {folder_path}..."):
                    start_time = time.time()
                    
                    # Get all image files from folder
                    valid_extensions = ('.png', '.jpg', '.jpeg')
                    image_files = [
                        f for f in os.listdir(folder_path) 
                        if f.lower().endswith(valid_extensions)
                    ]
                    
                    if not image_files:
                        st.warning("No valid images found (supported: .png, .jpg, .jpeg)")
                        st.stop()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, filename in enumerate(image_files):
                        status_text.text(f"Processing {i+1}/{len(image_files)}: {filename}")
                        progress_bar.progress((i+1)/len(image_files))
                        
                        try:
                            img_path = os.path.join(folder_path, filename)
                            key = os.path.splitext(filename)[0]
                            
                            # Check if already exists
                            existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()
                            
                            if key not in existing_ids:
                                # Process and add to database
                                embedding = embed_image(img_path)
                                collection.add(
                                    ids=[key],
                                    embeddings=[embedding],
                                    metadatas=[{"path": img_path}],
                                    documents=[key]
                                )
                                processed_files += 1
                                st.session_state.img_paths.append(img_path)
                            else:
                                skipped_files += 1
                                
                        except Exception as e:
                            st.error(f"Error processing {filename}: {str(e)}")
                            skipped_files += 1
                
                total_time = time.time() - start_time
                
                st.success(f"""
                Folder processing complete!
                - Processed: {processed_files} new images
                - Skipped: {skipped_files} (already exists or errors)
                - Total time: {total_time:.2f} seconds
                """)
                
            except Exception as e:
                st.error(f"Folder processing failed: {str(e)}")
        else:
            st.warning("Please enter a valid folder path")

# ========== NEW TAB 5: DELETE DATABASE ==========
with tab5:
    st.header("Database Management")
    
    st.warning("⚠️ This will permanently delete all images and embeddings!")
    
    if st.button("Delete All Data", type="primary"):
        try:
            with st.spinner("Clearing database..."):
                # Delete all items in the collection
                collection.delete(ids=collection.get()['ids'])
                
                # # Optionally delete the image files
                # for img_path in st.session_state.img_paths:
                #     print(img_path,"??????")
                #     try:
                #         os.remove(IMAGE_DIR)
                #     except:
                #         pass
                
                # Reset session state
                st.session_state.img_paths = []
                
            st.success("Database cleared successfully!")
            #st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Failed to delete database: {str(e)}")
    
    # Display database stats
    st.subheader("Current Database Status")
    if collection.count() > 0:
        st.info(f"""
        - Total images: {collection.count()}
        - Database path: {CHROMA_DB_PATH}
        - Last modified: {time.ctime(os.path.getmtime(CHROMA_DB_PATH))}
        """)
    else:
        st.info("Database is empty")

# Add some CSS for better display
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .stImage {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)