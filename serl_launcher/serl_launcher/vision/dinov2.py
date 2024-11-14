from transformers import AutoImageProcessor, FlaxDinov2ForImageClassification
from PIL import Image
import jax
import requests
import flax.linen as nn
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import vmap

from serl_launcher.vision.spatial import SpatialLearnedEmbeddings

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
    
def adapt_dinov2_model(model, input_channels=3, batch_size=96):
    """
    Adapt DINOv2 model to handle different input channels.
    
    Args:
        model: Original FlaxDinov2ForImageClassification model
        input_channels: Number of channels in your input data
        batch_size: expected batch size for processing
        
    Returns:
        Modified model with channel adaptation layer
    """
    # Get the expected number of channels from the model config
    expected_channels = model.config.num_channels  # Usually 3 for RGB
    
    # Initialize the channel adapter
    channel_adapter = ChannelAdapter(target_channels=expected_channels)
    
    # Create dummy input to initialize the adapter
    dummy_input = jnp.ones((batch_size, input_channels, 128, 128))  # Adjust size as needed
    adapter_params = channel_adapter.init(jax.random.PRNGKey(0), dummy_input)
    
    # Modified forward function
    @partial(jax.jit, static_argnums=(3,))
    def forward_batch(params, adapter_params, batch, train=False):
        # First apply channel adaptation
        adapted_batch = channel_adapter.apply(adapter_params, batch)
        # Then pass through the original model
        # breakpoint()
        output = model.__call__(adapted_batch,
                                params=params,
                                train=train,
                                output_hidden_states=True
                                )
        return output
    
    return forward_batch, adapter_params

class Dinov2ImageEncoder():
    def __init__(self, 
                 model_name: str = "facebook/dinov2-base-imagenet1k-1-layer", 
                 target_dim: int = 128,
                 pooling_method: str = "max"):
        self.model = FlaxDinov2ForImageClassification.from_pretrained(model_name, 
                                                                      from_pt=True, 
                                                                      output_hidden_states=True, 
                                                                      output_attentions=True)
        # self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.target_dim = target_dim
        self.pooling_method = pooling_method
        self.bottleneck_dim = 128

        self.pooling_fn = jax.jit(self._apply_pooling)

    def _apply_pooling(self, hidden_states):
        last_hidden_state = last_hidden_state.flatten()

        chunk_size = last_hidden_state.shape[0] // self.target_dim

        reshaped_vector = last_hidden_state[:chunk_size * self.target_dim].reshape((self.target_dim, chunk_size))

        # Apply max pooling to each chunk
        if self.pooling_method == "max":
            pooled_vector = jax.numpy.max(reshaped_vector, axis=1)  # Shape: (128,)
        elif self.pooling_method == "mean":
            pooled_vector = jax.numpy.mean(reshaped_vector, axis=1)
        # if self.pooling_method == "spatial_learned_embeddings":
        #     spatial_encoder = SpatialLearnedEmbeddings(height=128, width=128, channel=3, num_features=8)
        #     pooled_vector = spatial_encoder(last_hidden_state)
        else:
            raise ValueError(f"Pooling method {self.pooling_method} not supported.")
        # breakpoint()
        # if self.bottleneck_dim is not None:
        #     pooled_vector = nn.Dense(features=self.bottleneck_dim)(last_hidden_state)
        #     pooled_vector = nn.LayerNorm()(pooled_vector)
        #     pooled_vector = nn.tanh(pooled_vector)

        return pooled_vector

    def encode(self, observation):
        # inputs = self.image_processor(images=image, return_tensors="np")
        # print(observation.shape)
        if len(observations.shape) == 3:
            observations = observations[None, ...]
        observations = observations.astype(jnp.float32)

        adapted_model, adapter_params = adapt_dinov2_model(self.model, input_channels=observation.shape[1], batch_size=observation.shape[0])

        inputs = observation

        outputs = adapted_model(
            self.model.params,
            {'params': adapter_params},
            inputs,
            train=False,
        )
        # outputs = self.model(inputs)
        hidden_states = outputs.hidden_states
        
        last_hidden_state = hidden_states[-1] # Shape: (1, 1, 768)
        
        # Vectorize the pooling operation across the batch
        pooled_vectors = vmap(self.pooling_fn)(last_hidden_state)
        
        return pooled_vectors
    
    @nn.compact
    def __call__(self, observations, train=False, encode=False):
        assert observations.shape[-3:] == (128, 128, 3), f"Expected image shape (128, 128, 3), got {observations.shape[-3:]}"
        # breakpoint()
        if observations.shape == (128, 128, 3):
            observations = observations.reshape(1,3,128,128)
        else:
            a, b, c, d = observations.shape
            observations = observations.reshape(a, d, b, c)
            observations = observations.astype(np.float32)

            x = jnp.array([self.encode(observations[i].reshape(1,d,b,c)) for i in range(a)])
            return x

        observations = observations.astype(np.float32)
        x = self.encode(observations)
        return x