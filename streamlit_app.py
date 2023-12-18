import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import json
import math
import ast



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

def model(filter, input_sequence, input_unique):
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
    if filter == 'Avoid Barre':
        for i, r in recommend.iterrows():
            b = 0
            for c in input_unique:
                if feature_table[feature_table['chord']==c][feature_table['pattern']==r[c]]['barre_chord'].values[0] == 1:
                    b += 1
            if b!=0:
                recommend = recommend.drop([i])
        recommend = recommend.reset_index(drop=True)
        filter = 'can not '+filter
    elif filter == 'Lower Fret Only':
        for i, r in recommend.iterrows():
            b = 0
            for c in input_unique:
                if feature_table[feature_table['chord']==c][feature_table['pattern']==r[c]]['fret_start'].values[0] > 5:
                    b += 1
            if b!=0:
                recommend = recommend.drop([i])
        recommend = recommend.reset_index(drop=True)
        filter = 'can not play with '+filter
        filter += " (some chords require higher fret > 5)"
    elif filter == 'Less Fingers':
        finger = ['index','middle','ring','pinky']
        string_cols = [f+'_finger_string' for f in finger]
        for i, r in recommend.iterrows():
            b = 0
            for c in input_unique:
                if feature_table[feature_table['chord']==c][feature_table['pattern']==r[c]][string_cols].replace('',np.nan).count(axis=1).values[0] > 3:
                    b += 1
            if b!=0:
                recommend = recommend.drop([i])
        recommend = recommend.reset_index(drop=True)
        filter = 'can not play with '+filter
        filter += " (some chords require all 4 fingers)"
    if len(recommend) == 0:
        st.text('To play this song you '+filter.lower()+', sorry 😭🙏')
    else:
        regenerate(recommend, input_unique, finger_data, 1)
        re = 1
        if st.button("Regenerate"):
            re += 1
            regenerate(recommend, input_unique, finger_data, re)

def regenerate(recommend, input_unique, finger_data, re):
    rows = math.ceil(len(input_unique)/2)
    cols = 2 if len(input_unique)>1 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[chord_convert[i] for i in input_unique])
    fig.update_layout(height=300*rows)
    count = 0
    fig.update_yaxes(range=[0, 5], rangemode="tozero", tickvals=[0, 1, 2, 3, 4, 5], ticktext=['E', 'A', 'D', 'G', 'B', 'E'])
    fret = []
    for i in input_unique:
        count += 1
        row = math.ceil(count/2)
        col = (count+1)%2+1
        if re > 1:
            df = finger_data[np.multiply(finger_data['chord']==i,finger_data['pattern']==recommend.loc[recommend.head(re)[i].index[-1]].values[0]+1)]  
        else:
            df = finger_data[np.multiply(finger_data['chord']==i,finger_data['pattern']==recommend.head(1)[i].values[0]+1)]      
        if df['1'].values[0]!='':
            index_finger = ast.literal_eval(df['1'].values[0])
            fret.append(index_finger['fret'])
        else:
            index_finger = {'string':'','fret':''}
        if df['2'].values[0]!='':
            middle_finger = ast.literal_eval(df['2'].values[0])
            fret.append(middle_finger['fret'])
        else:
            middle_finger = {'string':'','fret':''}
        if df['3'].values[0]!='':
            ring_finger = ast.literal_eval(df['3'].values[0])
            fret.append(ring_finger['fret'])
        else:
            ring_finger = {'string':'','fret':''}
        if df['4'].values[0]!='':
            pinky_finger = ast.literal_eval(df['4'].values[0])
            fret.append(pinky_finger['fret'])
        else:
            pinky_finger = {'string':'','fret':''}

        plot_fretboard(fig, row, col, index_finger['string'], index_finger['fret'], middle_finger['string'], middle_finger['fret'], ring_finger['string'], ring_finger['fret'], pinky_finger['string'], pinky_finger['fret'])
    fig.update_xaxes(range=[min(fret)-1, max(fret)], tickmode='linear', tick0=1, dtick=1)
    text = '<span style="color: blue;">●</span>'
    text += " : Index Finger | "
    text += '<span style="color: purple;">●</span>'
    text += " : Middle Finger | "
    text += '<span style="color: orange;">●</span>'
    text += " : Ring Finger | "
    text += '<span style="color: yellow;">●</span>'
    text += " : Pinky Finger"
    st.markdown(text, unsafe_allow_html=True)
    st.plotly_chart(fig, theme=None, use_container_width=True)

def plot_fretboard(fig, row, col, index_string, index_fret, middle_string, middle_fret, ring_string, ring_fret, pinky_string, pinky_fret):
    data = []
    finger = []
    if index_string != '':
        if isinstance(index_string,list):
            for i in index_string:
                data.append([index_fret-0.5, abs(6-i)])
                finger.append(1)
        else:
            data.append([index_fret-0.5,  abs(6-index_string)])
            finger.append(1)
    if middle_string != '':
        if isinstance(middle_string,list):
            for i in middle_string:
                data.append([middle_fret-0.5, abs(6-i)])
                finger.append(2)
        else:
            data.append([middle_fret-0.5, abs(6-middle_string)])
            finger.append(2)
    if ring_string != '':
        if isinstance(ring_string,list):
            for i in ring_string:
                data.append([ring_fret-0.5, abs(6-i)])
                finger.append(3)
        else:
            data.append([ring_fret-0.5, abs(6-ring_string)])
            finger.append(3)
    if pinky_string != '':
        if isinstance(pinky_string,list):
            for i in pinky_string:
                data.append([pinky_fret-0.5, abs(6-i)])
                finger.append(4)
        else:
            data.append([pinky_fret-0.5, abs(6-pinky_string)])
            finger.append(4)

    df = pd.DataFrame(data, columns=['fret', 'string'])
    fig.add_trace(go.Scatter(
        x=df["fret"],
        y=df["string"],
        mode="markers",
        marker=dict(color=finger),
        marker_size=10,showlegend = False,hoverinfo='skip'
    ), row=row, col=col)    

def callback():
    st.session_state.button_clicked = True

st.title('ChordBrew 🎸')
st.text('A chord fingering generator by CN4')
st.text('-'*100)
input_sequence =  st.text_input('Input sequence of chords separated by space ex. G B C')
distinct_input = list(set(input_sequence))
error = ''

if "button_clicked" not in st.session_state:    
    st.session_state.button_clicked = False

filter = st.selectbox("Filter", ["Normal", "Avoid Barre", "Lower Fret Only", "Less Fingers"])

if (st.button("Run", type="primary", on_click=callback) or st.session_state.button_clicked):
    input_sequence = input_sequence.split()
    if len(input_sequence) > 2:
        convert_chord = json.load(open('convert_chord.json'))
        chord_convert = { convert_chord[k]:k for k in convert_chord}
        for i, c in enumerate(input_sequence):
            if c not in convert_chord:
                error = 'Sorry, our model accept only 144 common chords in these following formats: \n\n\tX (Xmajor), Xm (Xminor), X7 (Xdominant7), X5 (power chord),\n\tXdim (Xdiminished), Xdim7 (Xdiminished7), Xaug (Xaugmented),\n\tXsus2 (Xsuspended2), Xsus4 (Xsuspended4), X7 (Xdominant7),\n\tXm7 (Xminor7), X7sus4 (Xdominant7suspended4)'
                error += '\n\twhere X is C, C#, D, Eb, E, F, F#, G, G#, A, Bb, or B.'
                break
            else:
                input_sequence[i] = convert_chord[input_sequence[i]]
        if error == '':
            input_unique = list(set(input_sequence))
            if len(input_unique) > 1: 
                model(filter, input_sequence, input_unique)
            else:
                st.text("Please type a sequence of chords (at least 3 and more than 1 chord!) in above input box 👆")
        else:
            st.text(error)
    else:
        st.text("Please type a sequence of chords (at least 3) in above input box 👆")


st.text('-'*100)
st.text('This project is a part of CSS400 Project Development, academic year 2023/1.')
st.text('Developed, Designed, and Implemented by\n\tApinya\t\tSriyota\t\t\t6322771534\n\tSasikarn\tKhodphuwiang\t\t6322772375\n\tChanawong\tKaroon-ngampun\t\t6322774025')
