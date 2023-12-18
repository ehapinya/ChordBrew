import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



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

input_sequence =  st.text_input('Input sequence of chords here ex. G B C')
distinct_input = list(set(input_sequence))
error = ''

if st.button("Run"):
    for c in input_sequence:
        if c not in []:
            error = 'Sorry, we accept only 144 common chords in the following formats: \nX (Xmajor), Xm (Xminor), Xdominant7, X5, Xdiminished, Xdiminished7,\nXaugmented, Xsus2, Xsus4, X7 (Xmajor7), Xm7 (Xminor7), X7sus4'
    if error == '':
        finger_data = pd.read_excel('fingering_data.xlsx', keep_default_na=False, index_col=0)
        feature_table = pd.read_excel('feature_table.xlsx', keep_default_na=False, index_col=0)
    else:
        st.text(error)