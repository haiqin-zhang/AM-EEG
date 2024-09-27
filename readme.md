## Repository for EEG analysis in the audiomotor piano project.
All analyses are done in the eelbrain-cnsp environment, requirements in .txt file.

## Helpful files: 
individual_ERP_check: view the ERPs for different periods for each subject. Ensures that the preprocessing was done properly and that you can see the onset ERPs

## Steps:
1. Check file notes on AM_participants (google sheet)
2. Run pp_prepost_new.py (crops EEG recording into listen, motor, error files, each pre and post, and preprocesses each file)
3. Run ep_ERP.py in `pipelines_LM` folder to extract epochs and ERPs

---

