from preprocessing import preprocess_connect_4


def main():
    """
    Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc for the report
    :return:
    """
    print('Preprocessing...')
    filename_clean, filename_clean_enc = preprocess_connect_4.preprocess()
    print('Applying hierarchical clustering...')



if __name__ == '__main__':
    main()
