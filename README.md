# FaceRecognitionModel

- Install the libraries from `requirements.txt`
- Don't forget the model checkpoints!


## Functions

get_video_feature_embedding: 
Gets the average feature embedding for a video, reading from a byte array

        video: Video file data, as a byte array
        rate: Rate to sample frames, in seconds (default 0.25 seconds)

get_video_feature_embedding_filepath:
Gets the average feature embedding for a video, opening from a filepath

        video_path: Path to video file
        rate: Rate to sample frames, in seconds (default 0.25 seconds)
