from flask import *
from flask import Flask
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
import base64

app = Flask(__name__)

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        return dict(prediction)['displayNames']

# [END aiplatform_predict_image_classification_sample]

@app.route("/")
def main():
    return render_template("form.html")

@app.route('/success',methods = ["POST"])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save("image.jpg")
        diag1 = predict_image_classification_sample(
            project="599337888132",
            endpoint_id="5834189016886411264",
            location="us-central1",
            filename="image.jpg"
        )
        diag = "The diagnosis is: " + diag1[0]
        return render_template("Success.html",string_variable = diag)

if __name__ == '__main__':
    app.run(debug=True)
