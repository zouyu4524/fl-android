# Federated learning on Android

Implementation of federated learning on Android devices.

## Demo

<p align="center">
<img src="https://camo.githubusercontent.com/b3a3e6429ce42bb844cadd63abc65b15ace44d18/687474703a2f2f7777312e73696e61696d672e636e2f6c617267652f393364386637323167793167327a6778796b6261376a3230753031706f646d622e6a7067" alt="Screenshot of the Android app." width="300"></p>

## Project Structure

<pre>
fl-android
	|-- preprocess: Data preprocess, written in matlab 
	|-- client: Android client to perform on-device learning, written in android, built by Gradle
	|-- server: server to aggregate weights updated by clients, written in java, built by Gradle
	|-- README.md: readme
</pre>
## Illustration

We are going to implement the federated learning in realistic manner so that we can measure energy consumptions on real mobile devices. In particular, the training process happens at the **end devices (i.e., clients)**, such as mobile phones, tablets with their local stored or generated data. Then the updated weights of machine learning model are uploaded to a central **server**, who performs weights aggregation periodically, for example, averages the weights according to *FedAvg* algorithm. The updated model is then pushed to end users. The procedure goes on with higher model accuracy for end users and preserving their privacy.

## Installation & Running Instructions

### Prerequisite

* Gradle
* JDK
* Intellij IDEA (for sever project)
* Android Studio (for client project)
* Dropbox (for both server and clients)

Firstly, clone the repo as follows:  

```
git clone https://github.com/zouyu4524/fl-android.git
```

### Server-side

**Make sure Dropbox is installed and logged in on the server.**

Using Intellij IDEA to import the sever-side project.

`import project` &rarr; Locate `build.gradle` under `server/` &rarr; Click Ok &rarr; Leave all the settings default and click OK.

In Intellij IDEA, build the project first then you can run `HARClassifierNNAverageWeights` as one instance of model averaging.

Before you run `HARClassifierNNAverageWeights`, you should check three `final String` given at the top of Class, i.e., `updatedModel`, `originModel` and `onDeviceModelPath`:

* `updatedModel`: location of the updated model is stored
* `originModel`: location of the original model is stored
* `onDeviceModelPath`: path where on-device trained models stored, which should be `DropsyncFiles` under `Dropbox` in realistic deployment

**PS**: `updatedModel` and `originModel` should be the same one in realistic deployment and under `Dropbox/DropsyncFiles/`

### Client-side

On any participated android device, install [Dropbox](https://play.google.com/store/apps/details?id=com.dropbox.android&hl=en) and [DropSync](https://play.google.com/store/apps/details?id=com.ttxapps.dropsync) apps first.
By default, there should be a synchronizing folder created after DropSync installed, named as `DropsyncFiles`.  
Using Android Studio to import the client-side project.