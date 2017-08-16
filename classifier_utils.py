## Put the training data into matrices
def build_labels(train_ranges, positive_ranges):
    labels = []
    for i in range(0, len(train_ranges)):
        samples_in_range = train_ranges[i][1] - train_ranges[i][0]
        #print 'Samples in range =', samples_in_range
        
        if i in positive_ranges:
            for j in range(0, samples_in_range):
                labels.append(1)
        else:
            for j in range(0, samples_in_range):
                labels.append(0)
            

    return labels

def wanted_data(train_ranges):
    wanted = []

    for i in range(0, len(train_ranges)):
        samples_in_range = train_ranges[i][1] - train_ranges[i][0]
        
        for j in range(train_ranges[i][0], train_ranges[i][1]):
            wanted.append(j)

    return wanted

def take_row_ranges(train_ranges, array_2d):
    wanted = wanted_data(train_ranges)

    return array_2d[np.array(wanted)]
    
def build_training_data(train_ranges, positive_ranges, spec):
    train_labels = build_labels(train_ranges, positive_ranges)

    print '# of training labels = ', len(train_labels)

    train_vectors = take_row_ranges(train_ranges, spec)

    print '# of training vectors = ', train_vectors.shape[0]

    assert(train_vectors.shape[0] == len(train_labels))

    return train_vectors, train_labels

# Create prediction test function
def predict_and_score(all_square_lines, positive_range_inds, squareSpectrogram):
    Sq, sq = build_training_data(all_square_lines,
                                 positive_range_inds,
                                 squareSpectrogram)

    y_pred = gnbF.predict(Sq)
    print("Number of mislabeled points in undivided test set out of a total %d points : %d"
          % (Sq.shape[0],(sq != y_pred).sum()))

    print 'Precision score =', metrics.precision_score(sq, y_pred, [45, 90])
    print 'Recall score =', metrics.recall_score(sq, y_pred, [45, 90])

