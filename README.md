# Federated learning on Android

Implementation of federated learning on Android devices.

## Demo

![Screenshot of the Android app.](http://ww1.sinaimg.cn/large/93d8f721gy1g2zgxykba7j20u01podmb.jpg)

## Project Structure

<pre>
fl-android
	|-- preprocess: Data preprocess, written in matlab 
	|-- client: Android client to perform on-device learning, written in android, built by Gradle
	|-- server: server to aggregate weights updated by clients, written in java, built by Gradle
	|-- README.md: readme
</pre>
## Illsutration

We are going to implement the federated learning in realistic manner so that we can measure energy consumptions on real mobile devices. In particular, the training process happens at the **end devices (i.e., clients)**, such as mobile phones, tablets with their local stored or generated data. Then the updated weights of machine learning model are uploaded to a central **server**, who performs weights aggregation periodically, for example, averages the weights according to *FedAvg* algorithm. The updated model is then pushed to end users. The procedure goes on with higher model accuracy for end users and preserving their privacy.