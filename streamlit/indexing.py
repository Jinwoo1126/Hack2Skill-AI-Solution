from absl import app
from google.cloud import aiplatform
import base64
from google.cloud import storage
from google.protobuf import struct_pb2
from google.oauth2 import service_account
import typing
from variable import args


class EmbeddingResponse(typing.NamedTuple):
  text_embedding: typing.Sequence[float]
  image_embedding: typing.Sequence[float]


class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""
    def __init__(self, project : str, location : str = "asia-northeast3"):
        api_regional_endpoint = f"{location}-aiplatform.googleapis.com"
        client_options = {"api_endpoint": api_regional_endpoint}
        cred = service_account.Credentials.from_service_account_file(args.KEY_PATH)
        
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(credentials=cred, client_options=client_options)
        self.location = location
        self.project = project


    def get_embedding(self, text : str = None, image_bytes : bytes = None):
        if not text and not image_bytes:
            raise ValueError('At least one of text or image_bytes must be specified.')

        instance = struct_pb2.Struct()
        if text:
            instance.fields['text'].string_value = text

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields['image'].struct_value
            image_struct.fields['bytesBase64Encoded'].string_value = encoded_content

        instances = [instance]
        endpoint = (f"projects/{self.project}/locations/{self.location}"
        "/publishers/google/models/multimodalembedding@001")
        response = self.client.predict(endpoint=endpoint, instances=instances)

        text_embedding = None
        if text:    
            text_emb_value = response.predictions[0]['textEmbedding']
            text_embedding = [v for v in text_emb_value]

        image_embedding = None
        if image_bytes:    
            image_emb_value = response.predictions[0]['imageEmbedding']
            image_embedding = [v for v in image_emb_value]

        return EmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=image_embedding
        )

