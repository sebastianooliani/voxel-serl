from transformers import AutoImageProcessor, FlaxDinov2ForImageClassification
from PIL import Image
import jax
import requests

def test_dinov2():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    model = FlaxDinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer", from_pt=True, output_hidden_states=True, output_attentions=True)

    inputs = image_processor(images=image, return_tensors="np")
    outputs = model(**inputs)
    logits = outputs.logits
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions

    last_hidden_state = hidden_states[-1]
    last_hidden_state = last_hidden_state.flatten()

    target_dim=128

    chunk_size = last_hidden_state.shape[0] // target_dim

    reshaped_vector = last_hidden_state[:chunk_size * target_dim].reshape((target_dim, chunk_size))

    # Apply max pooling to each chunk
    pooled_vector = jax.numpy.max(reshaped_vector, axis=1)  # Shape: (128,)

    print(pooled_vector.shape)

    print(last_hidden_state.shape)

if __name__ == "__main__":
    test_dinov2()

class Dinov2ImageEncoder():
    def __init__(self, 
                 model_name: str = "facebook/dinov2-base-imagenet1k-1-layer", 
                 target_dim: int = 128,
                 pooling_method: str = "max"):
        self.model = FlaxDinov2ForImageClassification.from_pretrained(model_name, from_pt=True, output_hidden_states=True, output_attentions=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.target_dim = target_dim
        self.pooling_method = pooling_method

    def encode(self, image: Image):
        inputs = self.image_processor(images=image, return_tensors="np")
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states

        last_hidden_state = hidden_states[-1]
        last_hidden_state = last_hidden_state.flatten()

        chunk_size = last_hidden_state.shape[0] // self.target_dim

        reshaped_vector = last_hidden_state[:chunk_size * self.target_dim].reshape((self.target_dim, chunk_size))

        # Apply max pooling to each chunk
        if self.pooling_method == "max":
            pooled_vector = jax.numpy.max(reshaped_vector, axis=1)  # Shape: (128,)
        elif self.pooling_method == "mean":
            pooled_vector = jax.numpy.mean(reshaped_vector, axis=1)
        else:
            raise ValueError(f"Pooling method {self.pooling_method} not supported.")

        return pooled_vector