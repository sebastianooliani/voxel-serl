from transformers import AutoImageProcessor, FlaxDinov2ForImageClassification
from PIL import Image
import jax
import requests
import flax.linen as nn
import numpy as np
import jax.numpy as jnp

def test_dinov2():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer")
    model = FlaxDinov2ForImageClassification.from_pretrained("facebook/dinov2-base-imagenet1k-1-layer", from_pt=True, output_hidden_states=True, output_attentions=True)

    inputs = image_processor(images=image, return_tensors="np")
    print(inputs['pixel_values'].shape)
    print("\n")
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

    # print(pooled_vector.shape)

    # print(last_hidden_state.shape)

if __name__ == "__main__":
    test_dinov2()

class ChannelAdapter(nn.Module):
    target_channels: int
    
    @nn.compact
    def __call__(self, x):
        return nn.Conv(features=self.target_channels, 
                      kernel_size=(1, 1),
                      strides=(1, 1))(x)
    
def adapt_dinov2_model(model, input_channels=96):
    """
    Adapt DINOv2 model to handle different input channels.
    
    Args:
        model: Original FlaxDinov2ForImageClassification model
        input_channels: Number of channels in your input data
        
    Returns:
        Modified model with channel adaptation layer
    """
    # Get the expected number of channels from the model config
    expected_channels = model.config.num_channels  # Usually 3 for RGB
    
    # Initialize the channel adapter
    channel_adapter = ChannelAdapter(target_channels=expected_channels)
    
    # Create dummy input to initialize the adapter
    dummy_input = jnp.ones((input_channels, 1, 224, 224))  # Adjust size as needed
    adapter_params = channel_adapter.init(jax.random.PRNGKey(0), dummy_input)
    
    # Modified forward function
    def modified_forward(params, input_ids, **kwargs):
        # First apply channel adaptation
        adapted_input = channel_adapter.apply(adapter_params, input_ids)
        # Then pass through the original model
        return model.__call__(adapted_input, **kwargs)
    
    return modified_forward, adapter_params

class Dinov2ImageEncoder():
    def __init__(self, 
                 model_name: str = "facebook/dinov2-base-imagenet1k-1-layer", 
                 target_dim: int = 128,
                 pooling_method: str = "max"):
        self.model = FlaxDinov2ForImageClassification.from_pretrained(model_name, 
                                                                      from_pt=True, 
                                                                      output_hidden_states=True, 
                                                                      output_attentions=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.target_dim = target_dim
        self.pooling_method = pooling_method

    def encode(self, observation):
        # inputs = self.image_processor(images=image, return_tensors="np")
        adapted_model, adapter_params = adapt_dinov2_model(self.model, input_channels=observation.shape[0])

        inputs = observation

        outputs = adapted_model(
            {'params': adapter_params},
            inputs,
            train=False,
        )
        # outputs = self.model(inputs)
        hidden_states = adapted_model.hidden_states

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
    
    @nn.compact
    def __call__(self, observations, train=False, encode=False):
        assert observations.shape[-3:] == (128, 128, 3), f"Expected image shape (128, 128, 3), got {observations.shape[-3:]}"
        # breakpoint()
        if observations.shape == (128, 128, 3):
            observations = observations.reshape(1,3,128,128)
        else:
            pass

        observations = observations.astype(np.float32)
        x = self.encode(observations)
        return x