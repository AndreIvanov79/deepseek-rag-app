import os
import streamlit as st
import tempfile
from pathlib import Path
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import ModelLoader
from utils.rag_utils import RAGManager
from utils.chunking import ResponseChunker
from utils.image_utils import CodeImageProcessor

# Set page configuration
st.set_page_config(
    page_title="DeepSeek Coder RAG App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f7ff;
    }
    .chat-message.assistant {
        background-color: #f0f0f0;
    }
    .chat-message .avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .content {
        width: 100%;
    }
    /* Code style */
    pre {
        background-color: #f6f8fa !important;
        border-radius: 0.3rem;
        padding: 1rem;
    }
    code {
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Status indicators */
    .status-indicator {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .status-green {
        background-color: #28a745;
    }
    .status-red {
        background-color: #dc3545;
    }
    .status-yellow {
        background-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "selected_project" not in st.session_state:
    st.session_state.selected_project = None

if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

if "image_processor" not in st.session_state:
    st.session_state.image_processor = CodeImageProcessor()

# Create necessary directories
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/indices", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Sidebar
with st.sidebar:
    st.title("🧠 DeepSeek-Coder RAG")
    
    # Model loading section
    st.header("Model")
    
    model_status = "🟢 Loaded" if st.session_state.model_loaded else "🔴 Not Loaded"
    st.markdown(f"**Status:** {model_status}")
    
    if not st.session_state.model_loaded:
        if st.button("Load Model"):
            with st.spinner("Loading DeepSeek-Coder model..."):
                try:
                    # Initialize and load the model
                    if "model_loader" not in st.session_state:
                        st.session_state.model_loader = ModelLoader()
                    
                    st.session_state.model_loader.load_model()
                    st.session_state.model_loaded = True
                    
                    # Initialize RAG manager after model is loaded
                    if "rag_manager" not in st.session_state:
                        st.session_state.rag_manager = RAGManager(st.session_state.model_loader)
                    
                    st.success("Model loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    else:
        # Show model info
        if "model_loader" in st.session_state:
            model_info = st.session_state.model_loader.get_model_info()
            st.markdown(f"**Model:** {model_info.get('model_name', 'Unknown')}")
            st.markdown(f"**Device:** {model_info.get('device', 'Unknown')}")
            if 'memory_allocated_gb' in model_info:
                st.markdown(f"**Memory Usage:** {model_info.get('memory_allocated_gb', 0):.2f} GB")
            
        # Clear model from memory button
        if st.button("Unload Model"):
            if "model_loader" in st.session_state:
                del st.session_state.model_loader
            if "rag_manager" in st.session_state:
                del st.session_state.rag_manager
            st.session_state.model_loaded = False
            st.rerun()
    
    # Project management section
    st.header("Project Context")
    
    # Project upload
    if st.session_state.model_loaded:
        uploaded_file = st.file_uploader("Upload Project File", type=None, accept_multiple_files=False)
        project_name = st.text_input("Project Name")

        if uploaded_file is not None and project_name:
            if st.button("Process Project"):
                with st.spinner("Processing project..."):
                    try:
                        # Save the uploaded file to a temporary location
                        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.close()

                        # Create index from the file
                        project_id, doc_count = st.session_state.rag_manager.create_index_from_zip(
                            project_name, temp_file.name
                        )

                        # Clean up the temporary file
                        os.unlink(temp_file.name)

                        st.success(f"Project '{project_name}' processed successfully! Indexed {doc_count} documents.")

                        # Set as selected project
                        st.session_state.selected_project = project_id

                    except Exception as e:
                        st.error(f"Error processing project: {str(e)}")
        
        # Project selection
        if "rag_manager" in st.session_state:
            projects = st.session_state.rag_manager.get_projects()
            if projects:
                project_options = {pid: name for pid, name in projects.items()}
                selected_option = st.selectbox(
                    "Select Project",
                    options=list(project_options.keys()),
                    format_func=lambda x: project_options[x],
                    index=0 if st.session_state.selected_project is None else list(project_options.keys()).index(st.session_state.selected_project)
                )
                
                if selected_option != st.session_state.selected_project:
                    st.session_state.selected_project = selected_option
                    st.success(f"Project context activated: {project_options[selected_option]}")
                
                # Delete project button
                if st.button("Delete Selected Project"):
                    if st.session_state.rag_manager.delete_project(st.session_state.selected_project):
                        st.success(f"Project '{project_options[st.session_state.selected_project]}' deleted.")
                        st.session_state.selected_project = None
                        st.rerun()
    
    # About section
    st.header("About")
    st.markdown("""
    This app uses DeepSeek-Coder 33B model with RAG for context-aware coding assistance.
    
    - Load your project as a ZIP file
    - Chat with the model using your project context
    - Get code completion and assistance
    """)

# Main chat interface
st.title("💬 DeepSeek-Coder Assistant")

# Display project context status
if st.session_state.selected_project and "rag_manager" in st.session_state:
    projects = st.session_state.rag_manager.get_projects()
    st.markdown(f"🔍 Using project context: **{projects[st.session_state.selected_project]}**")
else:
    st.markdown("⚠️ No project context selected. Responses will be based on model knowledge only.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display images in the message if any
        if "image_path" in message:
            st.image(message["image_path"], caption="Uploaded Code Image")

# Image upload for code extraction
uploaded_image = st.file_uploader("Upload an image of code", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_image.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_image.getvalue())
        image_path = tmp_file.name

    # Display the uploaded image
    st.image(image_path, caption="Uploaded Code Image", use_column_width=True)

    # Process the image to extract code
    with st.spinner("Extracting code from image..."):
        result = st.session_state.image_processor.extract_code_from_image(image_path)

    if result["success"]:
        extracted_code = result["text"]
        language = result["detected_language"]

        # Display the extracted code
        st.markdown(f"**Detected Language:** {language}")
        st.code(extracted_code, language=language if language != 'text' else None)

        # Add action buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Send to Chat"):
                # Create a message with the extracted code
                code_message = f"I've extracted this code from an image:\n```{language}\n{extracted_code}\n```"
                st.session_state.messages.append({"role": "user", "content": code_message, "image_path": image_path})
                st.rerun()

        with col2:
            if st.button("Ask about this Code"):
                # Create a message asking about the extracted code
                query_message = f"Please analyze and explain this code:\n```{language}\n{extracted_code}\n```"

                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": query_message, "image_path": image_path})

                # Process response if model is loaded
                if st.session_state.model_loaded:
                    with st.spinner("Analyzing code..."):
                        try:
                            if st.session_state.selected_project:
                                # Use project context for response
                                response_data = st.session_state.rag_manager.query_project(
                                    st.session_state.selected_project, query_message
                                )
                                full_response = response_data["response"]
                            else:
                                # Basic prompt for the model without context
                                model, tokenizer = st.session_state.model_loader.get_model_and_tokenizer()

                                # DeepSeek Chat Template
                                messages = [
                                    {"role": "user", "content": query_message}
                                ]
                                chat_text = tokenizer.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )

                                # Generate response
                                inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
                                outputs = model.generate(
                                    inputs["input_ids"],
                                    max_new_tokens=2048,
                                    temperature=0.1,
                                    top_p=0.95,
                                    do_sample=True
                                )
                                full_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                            # Add assistant response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": full_response})

                        except Exception as e:
                            st.error(f"Error analyzing code: {str(e)}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Error analyzing code: {str(e)}"})
                else:
                    st.error("Please load the model first from the sidebar.")

                st.rerun()
    else:
        st.error(f"Failed to extract code: {result.get('error', 'Unknown error')}")

    # Clean up the temporary file
    try:
        # Keep the file if it's in a message
        if not any(message.get("image_path") == image_path for message in st.session_state.messages):
            os.unlink(image_path)
    except:
        pass

# Chat input
if prompt := st.chat_input("Ask me about your code..."):
    if not st.session_state.model_loaded:
        st.error("Please load the model first from the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                try:
                    if st.session_state.selected_project:
                        # Use project context for response
                        response_data = st.session_state.rag_manager.query_project(
                            st.session_state.selected_project, prompt
                        )
                        full_response = response_data["response"]
                    else:
                        # Basic prompt for the model without context
                        model, tokenizer = st.session_state.model_loader.get_model_and_tokenizer()

                        # DeepSeek Chat Template
                        messages = [
                            {"role": "user", "content": prompt}
                        ]
                        chat_text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )

                        # Generate response
                        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=2048,
                            temperature=0.1,
                            top_p=0.95,
                            do_sample=True
                        )
                        full_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                    # Display response
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    message_placeholder.error(f"Error generating response: {str(e)}")
                    full_response = f"Error: {str(e)}"

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Clear chat button
if st.button("Clear Chat"):
    # Clean up any image files
    for message in st.session_state.messages:
        if "image_path" in message:
            try:
                if os.path.exists(message["image_path"]):
                    os.unlink(message["image_path"])
            except:
                pass

    st.session_state.messages = []
    st.rerun()