from flask import Flask, render_template, request
import os
import pickle
import warnings
import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.mixture import *
from matplotlib import pyplot as plt
import numpy as np
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from pydub import AudioSegment
import subprocess
from moviepy.editor import AudioFileCl
import re
import whisper
import openai
import json
import requests
from requests.auth import HTTPBasicAuth

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World"

@app.route("/test", methods=['POST'])
def test():
    data = request.form['code']
    print(data)
    return "Hello World"

@app.route("/run", methods=['POST'])
def about():
    data = request.form['code']
    perform_action(data)
    return "Hello World"

def transcribe(file, key,labels, model="base", count=0):
    model = whisper.load_model(model)
    transcript = model.transcribe(os.path.join(key, file))
    return labels[count]+ ":" +transcript['text']
    

def request_summary(input_text, temperature=0.1, max_tokens=768):
    openai.api_key = "OPEN AI KEY"
    summary = ""
    for text_partition in input_text:
        if text_partition != '':
            response = openai.Completion.create(
                max_tokens=max_tokens,
                model="text-curie-001",
                prompt=summarize_prompt(text_partition),
                temperature=temperature,
            )
            summary = summary + str(response.choices[0].text)
    return summary


def request_action_items(input_text, temperature=0.1, max_tokens=768):
    openai.api_key = "OPEN AI KEY"
    summary = ""
    for text_partition in input_text:
        if text_partition != '':
            response = openai.Completion.create(
                max_tokens=max_tokens,
                model="text-curie-001",
                prompt=action_item_prompt(text_partition),
                temperature=temperature,
            )
            summary = summary + str(response.choices[0].text)
    return summary

def action_item_prompt(prompt):
    return """give me precisely the action points in this meeting? in as few lines as possible. 
    \"{}\"""".format(prompt)


# GPT3 methods
def summarize_prompt(prompt):
    return """Could you precisely get the summary of this meeting? in less than 10 points. 
    \"{}\"""".format(prompt)


def reformat_prompt(prompt):
    return """Could you reformat this text? 
    \"{}\"""".format(prompt)

# This code sample uses the 'requests' library:
# http://docs.python-requests.org


def create_page(html, title):
    url = "https://nihal7676.atlassian.net/wiki/api/v2/pages" 
    auth_token = 'AUTH TOKEN'
    auth = HTTPBasicAuth('puramnihal@gmail.com', auth_token)

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    page_html = html

    payload = json.dumps({
        "spaceId": "1769475",
        "status": "current",
        "title": title,
        "parentId": "1769705",
        "body": {
            "representation": "storage",
            "value": page_html,
        }
    })

    response = requests.request(
        "POST",
        url,
        data=payload,
        headers=headers,
        auth=auth
    )       

    print(json.dumps(json.loads(response.text),
        sort_keys=True, indent=4, separators=(",", ": ")))

def transcribe_all(uniq_dir,labels, model = 'base'):
    #creating expresion for better speed
    p = re.compile(r'\d+')
    whole_transcript = []
    parts = os.listdir(os.path.join(uniq_dir))
    #sorting parts list
    parts = sorted(parts, key=lambda s: int(p.search(s).group()))
    count = 0
    for file in parts:
        text = transcribe(file, uniq_dir,labels, model=model, count=count)
        whole_transcript.append(text)
        os.remove(os.path.join(uniq_dir, file))
        count += 1
    os.rmdir(os.path.join(uniq_dir))
    whole_transcript = " ".join(whole_transcript)
    return whole_transcript

def divide_into_parts(uniq_dir, times, input_audio_file):
    if uniq_dir not in os.listdir():
        os.mkdir(uniq_dir)
    try:
        audio = AudioFileClip(input_audio_file)
        timestamp = 0
        parts = []

        for timelength in times:
            parts.append(audio.subclip(timestamp, timestamp + timelength))
            timestamp += timelength

        for i, fragment in enumerate(parts):
            fragment.write_audiofile(os.path.join(uniq_dir, f'part_{i}.mp3'))
    except Exception as e:
        print(f"An error occurred: {e}")


def SegmentFrame(clust, segLen, frameRate, numFrames):
    frameClust = np.zeros(numFrames)
    for clustI in range(len(clust)-1):
        frameClust[clustI*segLen*frameRate:(clustI+1)*segLen*frameRate] = clust[clustI]*np.ones(segLen*frameRate)
    frameClust[(clustI+1)*segLen*frameRate:] = clust[clustI+1]*np.ones(numFrames-(clustI+1)*segLen*frameRate)
    return frameClust

def trainGMM(wavFile, frameRate, segLen, vad, numMix):
    wavData,_ = librosa.load(wavFile,sr=16000)
    mfcc = librosa.feature.mfcc(y=wavData, sr=16000, n_mfcc=20,hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad,:];
    print("Training GMM..")
    GMM = GaussianMixture(n_components=numMix,covariance_type='diag').fit(mfcc)
    var_floor = 1e-5
    segLikes = []
    segSize = frameRate*segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0])/(frameRate*segLen)))):
        startI = segI*segSize
        endI = (segI+1)*segSize
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0]-1
        if endI==startI:    # Reached the end of file
            break
        seg = mfcc[startI:endI,:]
        compLikes = np.sum(GMM.predict_proba(seg),0)
        segLikes.append(compLikes/seg.shape[0])
    print("Training Done")

    return np.asarray(segLikes)

def VoiceActivityDetection(wavData, frameRate):
        # uses the librosa library to compute short-term energy
        ste = librosa.feature.rms(y=wavData,hop_length=int(16000/frameRate)).T
        thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
        return (ste>thresh).astype('bool')


def speakerdiarisationdf(hyp, frameRate, wavFile):
    audioname=[]
    starttime=[]
    endtime=[]
    speakerlabel=[]
            
    spkrChangePoints = np.where(hyp[:-1] != hyp[1:])[0]
    if spkrChangePoints[0]!=0 and hyp[0]!=-1:
        spkrChangePoints = np.concatenate(([0],spkrChangePoints))
    spkrLabels = []    
    for spkrHomoSegI in range(len(spkrChangePoints)):
        spkrLabels.append(hyp[spkrChangePoints[spkrHomoSegI]+1])
    for spkrI,spkr in enumerate(spkrLabels[:-1]):
        if spkr!=-1:
            audioname.append(wavFile.split('/')[-1].split('.')[0]+".wav")
            starttime.append((spkrChangePoints[spkrI]+1)/float(frameRate))
            endtime.append((spkrChangePoints[spkrI+1]-spkrChangePoints[spkrI])/float(frameRate))
            speakerlabel.append("Speaker "+str(int(spkr)))
    if spkrLabels[-1]!=-1:
        audioname.append(wavFile.split('/')[-1].split('.')[0]+".wav")
        starttime.append(spkrChangePoints[-1]/float(frameRate))
        endtime.append((len(hyp) - spkrChangePoints[-1])/float(frameRate))
        speakerlabel.append("Speaker "+str(int(spkrLabels[-1])))
    #
    speakerdf=pd.DataFrame({"Audio":audioname,"starttime":starttime,"endtime":endtime,"speakerlabel":speakerlabel})
    
    spdatafinal=pd.DataFrame(columns=['Audio','SpeakerLabel','StartTime','EndTime'])
    i=0
    k=0
    j=0
    spfind=""
    stime=""
    etime=""
    for row in speakerdf.itertuples():
        if(i==0):
            spfind=row.speakerlabel
            stime=row.starttime
        else:
            if(spfind==row.speakerlabel):
                etime=row.starttime        
            else:
                spdatafinal.loc[k]=[wavFile.split('/')[-1].split('.')[0]+".wav",spfind,stime,row.starttime]
                k=k+1
                spfind=row.speakerlabel
                stime=row.starttime
        i=i+1
    spdatafinal.loc[k]=[wavFile.split('/')[-1].split('.')[0]+".wav",spfind,stime,etime]
    return spdatafinal
    
def partition_text_func(transcript, stop_word_num=180):
    # divide the transcript into words
    words = transcript.split()

    # Create an empty list to store partitioned text, string to store the words before the stop word and a counter
    partitioned_text = []
    before_stop_word = ""
    counter = 0
    # Iterate over the words in the string
    for word in words:
        # Check if you've reached the word you want to stop at
        if counter == stop_word_num:
            if word[-1] != '.':
                # After the values is reached append until you reach an end of sentence ('.')
                before_stop_word += word + " "
            else:
                before_stop_word += word + " "
                # and finally, add the whole sentence as an element of the partitioned text list
                partitioned_text.append(before_stop_word)
                # Reset before_stop_word to an empty string and counter set to 0
                before_stop_word = ""
                counter = 0
        else:
            # Append the current word to the new string
            before_stop_word += word + " "
            counter = counter + 1
    else:
        if before_stop_word != '':
            # catch the last remaining part of text into another element of the list
            partitioned_text.append(before_stop_word)

    # print(partitioned_text)
    return partitioned_text

def generate_html_summary_and_action_items(summary, action_items):
    action_items_list = action_items.split('\n\n')
    action_items_html = "<ul>\n"
    
    for item in action_items_list:
        if item.strip():  # Exclude empty lines
            action_items_html += f"<li>{item.strip()}</li>\n"
    
    action_items_html += "</ul>"

    html = f"""
    <html>
    <head>
        <title>Meeting Summary and Action Items</title>
    </head>
    <body>
        <h2>Meeting Summary</h2>
        <p>{summary}</p>
        
        <h2>Action Items</h2>
        {action_items_html}
    </body>
    </html>
    """

    return html

def perform_action(code):
    print("running the code now")
    # Replace with the path to the MP3 file you want to delete
    mp3_file = "output.mp3"
    # Check if the file exists before attempting to delete it
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print(f"{mp3_file} has been deleted.")
    else:
        print(f"{mp3_file} does not exist or cannot be deleted.")
        # Replace with the path to the MP3 file you want to delete
    mp3_file = "output2.mp3"

    # Check if the file exists before attempting to delete it
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print(f"{mp3_file} has been deleted.")
    else:
        print(f"{mp3_file} does not exist or cannot be deleted.")
        # Replace with the path to the MP3 file you want to delete
    mp3_file = "output.wav"

    # Check if the file exists before attempting to delete it
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print(f"{mp3_file} has been deleted.")
    else:
        print(f"{mp3_file} does not exist or cannot be deleted.")

    # Replace with your own values
    CREDENTIALS_FILE = 'Curious Arch.json'
    FILE_ID = code
    OUTPUT_MP3_FILE = 'output.mp3'

    # Authenticate using service account credentials
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    drive_service = build('drive', 'v3', credentials=credentials)

    # Download the MP4 file
    request = drive_service.files().get_media(fileId=FILE_ID)
    fh = io.FileIO(OUTPUT_MP3_FILE, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%")

    print("MP4 downloaded successfully")

    input_file = OUTPUT_MP3_FILE
    output_file = "output2.mp3"

    # Run FFmpeg to convert the input MP4 to MP3
    ffmpeg_command = f'ffmpeg -i {input_file} -vn -ar 44100 -ac 2 -ab 192k -f mp3 {output_file}'
    subprocess.call(ffmpeg_command, shell=True)

    print("MP3 conversion complete")
    # Replace with the path to your input MP3 file
    mp3_file = "output2.mp3"

    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)

    # Specify the output WAV file
    wav_file = "output.wav"

    # Export the audio as WAV
    audio.export(wav_file, format="wav")

    print(f"Conversion from MP3 to WAV completed: {wav_file}")

    # Replace with the path to the MP3 file you want to delete
    mp3_file = "output2.mp3"

    # Check if the file exists before attempting to delete it
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print(f"{mp3_file} has been deleted.")
    else:
        print(f"{mp3_file} does not exist or cannot be deleted.")

    mp3_file = "output.mp3"

    # Check if the file exists before attempting to delete it
    if os.path.exists(mp3_file):
        os.remove(mp3_file)
        print(f"{mp3_file} has been deleted.")
    else:
        print(f"{mp3_file} does not exist or cannot be deleted.")

    segLen,frameRate,numMix = 3,50,128

    wavFile="output.wav"

    wavData,_ = librosa.load(wavFile,sr=16000)

    vad=VoiceActivityDetection(wavData,frameRate)

    mfcc = librosa.feature.mfcc(y=wavData, sr=16000, n_mfcc=20,hop_length=int(16000/frameRate)).T

    vad = np.reshape(vad,(len(vad),))

    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]

    mfcc = mfcc[vad,:];

    # n_components = np.arange(1, 25)
    # models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(mfcc)
    #         for n in n_components]

    clusterset = trainGMM(
    wavFile, frameRate, segLen, vad, numMix
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clusterset)  
    # Normalizing the data so that the data approximately 
    # follows a Gaussian distribution
    X_normalized = normalize(X_scaled)

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward') 
    clust=cluster.fit_predict(X_normalized)

    frameClust = SegmentFrame(clust, segLen, frameRate, mfcc.shape[0])

    pass1hyp = -1*np.ones(len(vad))

    pass1hyp[vad] = frameClust

    spkdf=speakerdiarisationdf(pass1hyp, frameRate, wavFile)
    spkdf["TimeSeconds"]=spkdf.EndTime-spkdf.StartTime

    input_audio_file = 'output.wav'
    uniq_dir = "parts"
    # Replace with the actual time durations you want
    divide_into_parts(uniq_dir, spkdf["TimeSeconds"], input_audio_file)
    print("Transcribing the files")

    final_transcript = transcribe_all(uniq_dir="parts", labels=spkdf["SpeakerLabel"])
    partition_text = partition_text_func(final_transcript)

    final_input = ""
    for sentence in partition_text:
        final_input = final_input + sentence

    summary = request_summary(partition_text)
    actions_items = request_action_items(partition_text)
    html_content = generate_html_summary_and_action_items(summary, actions_items)
    create_page(html_content, "Meeting Summary")

if __name__ == "__main__":
    app.run(host='0.0.0.0',ssl_context=('cert.pem', 'key.pem'))
