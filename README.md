# language_identification for Vichar app videos

1. Clone this repository
2. Please install the dependancies specified in the requirements.txt file as follows
```
pip install -r requirements.txt
```
3. Install youtube-dl and ffmpeg as follows
```
sudo apt-get install ffmpeg
sudo apt-get install youtube-dl
```
4. Download the model checkpoint provided in the drive
5. Given a downloadable video link you can predict the language as follows
```
import inference.prediction as predict_lang
language = predict_lang(video_url)
```
