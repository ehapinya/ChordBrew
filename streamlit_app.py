import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json
import math



def transition_table(prev_chord,next_chord):
    new_prev_chord = prev_chord.copy()
    new_next_chord = next_chord.copy()
    empty_prev = []
    for col in prev_chord:
        if prev_chord[col].values[0] == '':
            empty_prev.append(col)
    empty_next = []
    for col in next_chord:
        if sum(next_chord[col]=='') != 0:
            empty_next.append(col)
        elif sum(next_chord[col]=='') == len(next_chord):
            new_prev_chord = new_prev_chord.drop(columns=[col])
            new_next_chord = new_next_chord.drop(columns=[col])
    empty_common = list(set(empty_prev).intersection(empty_next))
    empty_prev = list(set(empty_prev) - set(empty_common))
    empty_next = list(set(empty_next) - set(empty_common))
    for col in empty_common:
        new_prev_chord[col] = new_prev_chord[col].replace('',new_next_chord[col].replace('',np.nan).mean())
        new_next_chord[col] = new_next_chord[col].replace('',new_next_chord[col].replace('',np.nan).mean())
    for col in empty_prev:
        new_prev_chord[col] = new_prev_chord[col].replace('',new_next_chord[col].replace('',np.nan).mean())
    for col in empty_next:
        new_next_chord[col] = new_next_chord[col].replace('',new_prev_chord[col].values[0])
    return new_prev_chord, new_next_chord

def model(input_sequence):
    finger_data = pd.read_excel('fingering_data.xlsx', keep_default_na=False, index_col=0)
    feature_table = pd.read_excel('feature_table.xlsx', keep_default_na=False, index_col=0)
    transition = []
    prev = input_sequence[0]
    for i in range(1,len(input_sequence)):
        next = input_sequence[i]
        if prev == next:
            continue
        if prev+' '+next in transition:
            prev = next
            continue
        transition.append(prev+' '+next)
        prev = next
    chords = {}
    input_unique = list(set(input_sequence))
    for c in input_unique:
        chords[c] = feature_table[feature_table['chord']==c].iloc[:,2:].reset_index(drop=True)
    transition_original = transition.copy()
    results = {}
    for i in range(len(transition_original)):
        transition = transition_original[i:]+transition_original[:i]
        transition_score = {}
        prev = transition[0].split()[0]
        next = transition[0].split()[1]
        for pattern in range(len(chords[prev])):
            current_score = 1000
            transition_score[pattern] = {'pattern':{prev:int(pattern)}}
            prev_chord = chords[prev].loc[[pattern]].copy()
            next_chord = chords[next].copy()
            prev_chord_processed, next_chord_processed = transition_table(prev_chord, next_chord)
            prev_chord_processed, next_chord_processed = prev_chord_processed.dropna(axis=1), next_chord_processed.dropna(axis=1)
            kmeans = KMeans(n_clusters=len(next_chord_processed)) 
            clusters = kmeans.fit_predict(next_chord_processed.dropna(axis=1).values)
            next_chord_processed['cluster'] = clusters
            result = next_chord_processed[next_chord_processed['cluster'] == kmeans.predict(prev_chord_processed.values[0].reshape(1, -1))[0]].iloc[:,:-1]
            score = math.dist(result.values[0],prev_chord_processed.values[0])
            if score < current_score:
                current_score = score
                next_result = result.index[0]
            transition_score[pattern]['pattern'][next] = next_result
            transition_score[pattern]['score'] = [current_score]
        for pair in transition[1:]:
            prev = pair.split()[0]
            next = pair.split()[1]
            for pattern in transition_score:
                try:
                    prev_pattern = transition_score[pattern]['pattern'][prev]
                    current_score = 1000
                    prev_chord = chords[prev].loc[[int(prev_pattern)]].copy()
                    if next in transition_score[pattern]['pattern']:
                        next_pattern = transition_score[pattern]['pattern'][next]
                        next_chord = chords[next].loc[[int(next_pattern)]].copy()
                    else:
                        next_chord = chords[next].copy()
                    prev_chord_processed, next_chord_processed = transition_table(prev_chord, next_chord)
                    prev_chord_processed, next_chord_processed = prev_chord_processed.dropna(axis=1), next_chord_processed.dropna(axis=1)
                    kmeans = KMeans(n_clusters=len(next_chord_processed)) 
                    clusters = kmeans.fit_predict(next_chord_processed.values)
                    next_chord_processed['cluster'] = clusters
                    result = next_chord_processed[next_chord_processed['cluster'] == kmeans.predict(prev_chord_processed.values[0].reshape(1, -1))[0]].iloc[:,:-1]
                    score = math.dist(result.values[0],prev_chord_processed.values[0])
                    if score < current_score:
                        current_score = score
                        next_result = result.index[0]
                    transition_score[pattern]['pattern'][next] = next_result
                    transition_score[pattern]['score'].append(current_score)
                except:
                    break
            results[i] = transition_score
    result = []
    row = []
    for i in results:
        for j in results[i]:
            res = results[i][j]
            result.append(res)
            try:
                row.append(list(map(lambda x: res['pattern'][x],input_unique))+[np.mean(res['score'])])
            except:
                continue
    recommend = pd.DataFrame(row, columns=input_unique+['score']).drop_duplicates().sort_values('score')
    for i in input_unique:
        st.dataframe(finger_data[np.multiply(finger_data['chord']==i,finger_data['pattern']==recommend.head(1)[i].values[0]+1)])
    
st.title('ChordBrew 🎸')
st.text('A chord fingering generator by CN4')
st.text('-'*100)
input_sequence =  st.text_input('Input sequence of chords separated by space ex. G B C')
distinct_input = list(set(input_sequence))
error = ''

if st.button("Run"):
    input_sequence = input_sequence.split()
    if len(input_sequence) != 0:
        convert_chord = json.load(open('convert_chord.json'))
        for i, c in enumerate(input_sequence):
            if c not in convert_chord:
                error = 'Sorry, our model accept only 144 common chords in these following formats: \n\n\tX (Xmajor), Xm (Xminor), X7 (Xdominant7), X5 (power chord),\n\tXdim (Xdiminished), Xdim7 (Xdiminished7), Xaug (Xaugmented),\n\tXsus2 (Xsuspended2), Xsus4 (Xsuspended4), X7 (Xdominant7),\n\tXm7 (Xminor7), X7sus4 (Xdominant7suspended4)'
                error += '\n\twhere X is C, C#, D, Eb, E, F, F#, G, G#, A, Bb, or B.'
                break
            else:
                input_sequence[i] = convert_chord[input_sequence[i]]
        if error == '':
            model(input_sequence)
        else:
            st.text(error)
    else:
        st.text("Please type a sequence of chords in above input box 👆")

st.text('-'*100)
st.text('This project is a part of CSS400 Project Development, academic year 2023/1.')
st.text('Developed, Designed, and Implemented by\n\tApinya\t\tSriyota\t\t\t6322771534\n\tSasikarn\tKhodphuwiang\t\t6322772375\n\tChanawong\tKaroon-ngampun\t\t6322774025')
