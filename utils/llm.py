import json
import requests

def chat(messages, model='tinydolphin', verbose=False):
    """
    Sends messages to a chat API and returns the response.
    
    Args:
        messages (list): A list of dictionaries representing the chat history.
        model (str): The model name to be used in the chat API. Defaults to 'tinydolphin'.
        verbose (bool): If True, prints each message content as it's received.

    Returns:
        dict: A dictionary containing the final message after all parts are received.
    """
    # Sending POST request to the chat API
    response = requests.post(
        "http://0.0.0.0:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
    )
    # Check for successful response status
    response.raise_for_status()

    # Initialize empty output string to accumulate message contents
    output = ""

    # Process each line in the response stream
    for line in response.iter_lines():
        # Convert line from JSON format to dictionary
        body = json.loads(line)
        # Raise an exception if there is an error in the response
        if "error" in body:
            raise Exception(body["error"])
        # Check if the stream is not yet done
        if not body.get("done"):
            # Get the message content and append to output string
            content = body.get("message", {}).get("content", "")
            output += content
            if verbose:
                print(content, end="", flush=True)
        # If the stream is marked as done, return the accumulated message
        if body.get("done"):
            body["message"]["content"] = output
            return body["message"]

def embed(message, model='nomic-embed-text'):
    """
    Sends a message to an embedding API and returns the response.

    Args:
        message (str): The message to be embedded.
        model (str): The embedding model name. Defaults to 'nomic-embed-text'.

    Returns:
        set: The embedding result as a set if successful, otherwise an error message.
    """
    url = "http://localhost:11434/api/embeddings"
    headers = {"Content-Type": "application/json"}
    
    # Sending POST request to the embedding API
    response = requests.post(url, json={"model": model, "prompt": message}, headers=headers)
    
    # Check if response is successful
    if response.status_code == 200:
        response_data = response.json()
        # Assuming the embedding is returned under the key 'embedding'
        embedding = response_data.get('embedding')
        if embedding:
            # Convert the embedding list to a set to remove duplicates and return
            return list(set(embedding))
        else:
            return "Error: Embedding not found in response"
    else:
        return f"Error: {response.status_code}"
    
def main():
    """
    Runs a command-line interface (CLI) chat bot using a local model.
    """
    messages = []

    while True:
        user_input = input("Enter a prompt: ").strip()
        # Exit the loop if input is empty
        if not user_input:
            break
        messages.append({"role": "user", "content": user_input})
        response_message = chat(messages, verbose=True)
        messages.append(response_message)
        print("\n\n")

if __name__ == "__main__":
    main()
