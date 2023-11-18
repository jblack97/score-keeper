# SCORE-KEEPER
This project aims to provide an end-to-end pipeline for validating the results of bets shown in an image of a betslip. This problem contains elements of OCR (extracting the text from screenshots of betslips), NLP (parsing the OCR output into distinct bets), and webscraping (validating the results of the bets).

This is still a work in progress. The OCR solution works well for screenshots of betslips, and GPT-3.5 can solve the NLP task with only a handful examples. I am currently working on generating a synthetic dataset to train my own tagger model to remove the OpenAI dependency, the code for this can be found in **src/bet_identification/betslip_tagger**.