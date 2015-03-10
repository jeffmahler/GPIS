import os
import sys
import testing_class
top='/mnt/terastation/shape_data/Cat50_ModelDatabase/'

K=5
alphas=[0.01, 0.1, 1, 10, 100, 1000]


category_list=os.listdir(top)
for category in category_list:
	if category.find('test_results')!=-1:
		writefile=open(top+"test_results/"+category+"_data_5_neighbors.txt","w")
		temp=top+category
		newtest=testing_class.testing_suite()
		newtest.adddir(temp)
		numfiles=len(newtest.all_files_)
		numtest=numfiles*1//4
		numtrain=numfiles*3//4
		newtest.make_train_test(numtrain,numtest)
		accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_PCA_tests(K)
		EVR=newtest.get_explained_variance_ratio()
		writefile.write("PCA TESTS\n")
		writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))
		writefile.write("\nexplained variance ratio:"+str(EVR))


		accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_FA_tests(K)
		writefile.write("\nFACTOR ANALYSIS TESTS\n")
		writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

		accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
		writefile.write("\nKernelPCA TESTS\n")
		writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

		accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
		writefile.write("\nFAST ICA TESTS\n")
		writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

		accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K)
		writefile.write("\nFAST ICA TESTS\n")
		writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

		for value in alphas:
			accuracy, results, recall, tnr, precision,npv,fpr=newtest.perform_KPCA_tests(K,value)
			writefile.write("\nDICTIONARY ALPHA="+value+" TESTS\n")
			writefile.write("accuracy:"+str(accuracy)+"\nresults:"+str(results)+"\nrecall:"+str(recall)+"\ntnr:"+str(tnr)+"\nprecision:"+str(precision)+"\nnpv:"+str(npv)+"\nfpr:"+str(fpr))

		writefile.close()

		with open(top+"test_results/"+category+"_data_5_neighbors.pk1", 'wb') as output:
        	pickle.dump(newtest, output, pickle.HIGHEST_PROTOCOL)
