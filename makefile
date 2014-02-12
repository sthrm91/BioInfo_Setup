proc:	proc.o svm.o model_training_testing.o feature_selection.o
	g++ -o proc proc.o svm.o model_training_testing.o feature_selection.o -pthread

proc.o:	main.cpp parallel.h
	g++ -c -o proc.o main.cpp

svm.o:	svm.cpp svm.h
	g++ -c -o svm.o svm.cpp

model_training_testing.o:	model_training_testing.cpp common.h
	g++ -c -o model_training_testing.o model_training_testing.cpp

feature_selection.o:	feature_selection.cpp common.h
	g++ -c -o feature_selection.o feature_selection.cpp

make clean:
	rm *.o proc
