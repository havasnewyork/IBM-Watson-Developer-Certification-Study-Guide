All content is from taken from the IBM Bluemix Watson Developer Cloud Documentation.

# Overview
The IBM® Speech to Text service provides an Application Programming Interface (API) that lets you add speech transcription capabilities to your applications. To transcribe the human voice accurately, the service leverages machine intelligence to combine information about grammar and language structure with knowledge of the composition of the audio signal. The service continuously returns and retroactively updates the transcription as more speech is heard.
* 3 interfaces: REST HTTP, Aync HTTP, Websocket API
* Supports up to 100MB of data per payload

## Websocket
* Single-socket, full duplex communication
* results are asynchronous in a single channel
* Reduced latency
* Reduced network utilization than other API

Return codes:
* 1000 -> normal closure
* 1002 -> protocol error
* 1006 -> closed abnormally
* 1009 -> frame size exceeded 4MB limit
* 1011 -> unexpected error

## HTTP REST
* Requires authentication in each call
* Requires 4 distinct requests in order to make a connection to server, incurring latency

Within HTTP REST, there are two types of requests: *Session* requests and *Sessionless* requests.

**Session Requests**: Enables client to create long, multi-turn/parallel conversations. You create a session by calling the `POST sessions` method. Example:
```
curl -X POST -u {username}:{password}
--cookie-jar cookies.txt
"https://stream.watsonplatform.net/speech-to-text/api/v1/sessions"
```

**Sessionless Requests**: Approppiate for batch processing but not for live speech recognition. You call `POST recognize` API for using the service. Example:
```
curl -X POST -u {username}:{password}
--header "Content-Type: audio/flac"
--data-binary @{path}audio-file.flac
"https://stream.watsonplatform.net/speech-to-text/api/v1/recognize"
```

## Asynchronous HTTP
You can register a callback URL to be notified by the service of the job status. You call the `POST register_callback` method and pass a callback URL and, optionally, a user-specified secret. In the following example, the user secret used is `ThisIsMySecret`:
```
curl -X POST -u {username}:{password}
"https://stream.watsonplatform.net/speech-to-text/api/v1/register_callback?callback_url=http://{user_callback_path}/results&user_secret=ThisIsMySecret"
```

## API Details
* Maximum alternatives: Accepts an `Int` value that tells the service to return the n-best alternative hypothesis.
* Interim Results: Intermediate hypotheses of a transcription that are likely to change before the service returns it's final results.
