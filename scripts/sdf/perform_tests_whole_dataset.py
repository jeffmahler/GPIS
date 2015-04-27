import IPython
import os
import sys
import testing_class

top = '/mnt/terastation/shape_data/Cat50_ModelDatabase/'
K = 5
test_percent = 0.25
alphas = [0.01, 0.1, 1, 10, 100, 1000]

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc > 1:
        K = int(sys.argv[1])
    if argc > 2:
        test_percent = float(sys.argv[2])

    # create testing object
    newtest = testing_class.testing_suite()
    newtest.adddir(top)

    # number of files
    numfiles = len(newtest.all_files_)
    numtest = int(numfiles * test_percent)
    numtrain = numfiles - numtest
    newtest.make_train_test(numtrain, numtest)

    # test PCA
    print 'Running PCA'
    accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
    EVR = newtest.get_explained_variance_ratio()

    exit()

    category_list = os.listdir(top)
    for category in category_list:
        if category.find('test_results')!=-1:
            # create result file
            writefile = open(top+"test_results/"+category+"_data_5_neighbors.txt","w")
            temp = top + category
            newtest = testing_class.testing_suite()
            newtest.adddir(temp)

            # number of files
            numfiles = len(newtest.all_files_)
            numtest = numfiles * test_percent
            numtrain = numfiles - numtest
            newtest.make_train_test(numtrain, numtest)

            # test PCA
            accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_PCA_tests(K)
            EVR = newtest.get_explained_variance_ratio()
            writefile.write("PCA TESTS\n")
            writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))
            writefile.write("\nexplained variance ratio:"+str(EVR))
                      
            # test factor analysis
            accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_FA_tests(K)
            writefile.write("\nFACTOR ANALYSIS TESTS\n")
            writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

            # test kernel PCA
            accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
            writefile.write("\nKernelPCA TESTS\n")
            writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

            '''
            accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
            writefile.write("\nFAST ICA TESTS\n")
            writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))
                        
            # test fast ICA
            accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
            writefile.write("\nFAST ICA TESTS\n")
            writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))
                        
            # test dictionary learning
            for value in alphas:
                accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K,value)
                writefile.write("\nDICTIONARY ALPHA="+value+" TESTS\n")
                writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

            '''
            writefile.close()

            # write model to pickle file
            with open(top+"test_results/"+category+"_data_5_neighbors.pk1", 'wb') as output:
                pickle.dump(newtest, output, pickle.HIGHEST_PROTOCOL)
