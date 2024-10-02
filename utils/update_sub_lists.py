import os
from scipy.io import loadmat, savemat

all_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

good_listen_subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
good_motor_subjects = ['01', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'] #double check 21


nonmusicians = ['01', '03', '04', '05', '08', '09', '10', '11', '16', '19', '20']
musicians = ['02', '06', '07', '12', '13', '14', '15', '17', '18', '21']

savemat('subject_lists.mat', 
        {
            'good_listen_subjects':good_listen_subjects,
            'good_motor_subjects': good_motor_subjects,
            'nonmusicians': nonmusicians,
            'musicians':musicians

        })

def load_subject_lists():
    subject_lists = loadmat('/Users/cindyzhang/Documents/M2/Audiomotor_Piano/AM-EEG/utils/subject_lists.mat')
    good_listen_subjects = subject_lists['good_listen_subjects'].tolist()
    good_motor_subjects = subject_lists['good_motor_subjects'].tolist()
    musicians = subject_lists['musicians'].tolist()
    nonmusicians = subject_lists['nonmusicians'].tolist()

    return good_listen_subjects, good_motor_subjects, musicians, nonmusicians