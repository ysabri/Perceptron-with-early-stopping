import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.math.*;


public class Preceptron {
	static final double LEARNING_RATE = 0.1;
	static int trainC, testC, tuneC; 
	static double threshold = 0;
	static int featNumber = -1;
	static String tuneSetName = null;
	static boolean tuneCountSet = false;

	public static void main(String[] args) {
		//check if input is in correct format
		if(args.length != 3) {
			System.out.println("Please enter the trainingFile name followed"
					+ "by the tuningFile name then the testFile name");
			System.exit(1);
		}

		//declare scanners for each file
		Scanner[] files = new Scanner[3];

		//check if files exist
		try {
			files[0] = new Scanner(new File(args[0]));
			files[1] = new Scanner(new File(args[1]));
			files[2] = new Scanner(new File(args[2]));
		} catch(FileNotFoundException e) {
			System.err.println("Could not find file '" + args[0] + "'.");
			System.exit(1);
		}

		int featTrain, featTune, featTest;
		tuneSetName = args[1];
		featTrain = findFeatsNumber(files[0]);
		featTune = findFeatsNumber(files[1]);
		featTest = findFeatsNumber(files[2]);
		//check if all feature numbers are the same
		if(featTrain!=featTune && featTune!=featTest){
			System.out.println("The number of features is different between"
					+ "the files");
			System.exit(1);
		}
		else
			featNumber = featTrain;


		//get the two values for each feature
		String[] trainFeats = findFeatsVals(files[0]);
		String[] testFeats = findFeatsVals(files[2]);

		//get the two output values
		String trainOutput = findOutputs(files[0]);
		String testOutput = findOutputs(files[2]);

		//get the item count for each file
		int trainItemCount = findItemCount(files[0]);
		int testItemCount = findItemCount(files[2]);

		//set the global counters
		trainC = trainItemCount;
		testC = testItemCount;

		//declare the weights and initialize all of them to zero
		double[] weights = new double[featNumber+1];
		Arrays.fill(weights, 0);

		//declare algorithm necessary variables
		double localErr, globalErr, tempOutput, realOutput, v=90000, tempV=0;
		int itr=0, stepSize = 10, patience = 5, j=0, i=0;
		double[] finalWeights = weights;
		int tempI= i;
		String currLine = null;
		//String tuneLine = "first";

		//move the scanner to the first input in train file
		while(files[0].hasNext() && currLine==null){
			String temp = files[0].nextLine();
			if(temp.equals("") || temp.startsWith("//"))
				continue;
			else
				currLine = temp;
		}

		while(j<patience){
			for(int k=0; k<stepSize; k++){
				double[] currFeatures = parseFeatures(currLine, trainFeats);
				realOutput = parseOutput(currLine, trainOutput);
				tempOutput = computePrediction(weights, currFeatures);
				localErr = realOutput - tempOutput;
				updateWeights(localErr, weights, currFeatures);
				if(files[0].hasNext() && !files[0].equals(""))
					currLine = files[0].nextLine();
			}
			i += stepSize;
			tempV = findErr(weights);
			if(tempV<v){
				j=0;
				finalWeights = weights;
				tempI = i;
				v = tempV;
			}
			else 
				j=j+1;
		}

		System.out.println("Error at the end of training: " + (tempV)*100);

		//get to the first line of inputs in test file
		currLine=null;
		while(files[2].hasNext() && currLine==null){
			String temp = files[2].nextLine();
			if(temp.equals("") || temp.startsWith("//"))
				continue;
			else
				currLine = temp;
		}
		//run the test file through the perceptron
		globalErr = 0;
		itr=0;
		do {
			double[] currFeatures = parseFeatures(currLine, testFeats);
			realOutput = parseOutput(currLine, testOutput);
			tempOutput = computePrediction(weights, currFeatures);
			localErr = realOutput - tempOutput;

			globalErr += Math.pow(localErr,2);
			
			if(testItemCount-1>itr)
				currLine = files[2].nextLine();
			
			itr++;
		} while(testItemCount>itr);

		System.out.println("Error measured on the test set: " + ((globalErr/itr)));

	}
	//compute the error on the tuning set 
	static double findErr(double[] weights){

		Scanner file = null;
		try {
			file = new Scanner(new File(tuneSetName));
		} catch (FileNotFoundException e) {
			System.out.println("This program just divided by zero");
		}
		//initialize the file 
		int featTune = findFeatsNumber(file);
		String[] feats = findFeatsVals(file);
		String outputs = findOutputs(file);
		int count = findItemCount(file);
		double glErr = 0, locErr, labelOutput, prediction;
		int itr=0;
		String line = null;
		//get to the first line of examples
		while(file.hasNext() && line==null){
			String temp = file.nextLine();
			if(temp.equals("") || temp.startsWith("//"))
				continue;
			else
				line = temp;
		}
		//compute the whole set
		do {
			
			if(line.equals(""))
				continue;
			
			glErr = 0;
			double[] currFeatures = parseFeatures(line, feats);
			labelOutput = parseOutput(line, outputs);
			prediction = computePrediction(weights, currFeatures);
			locErr = labelOutput - prediction;
			glErr += Math.pow(locErr,2);
			
			if(count-1>itr)
				line = file.nextLine();
			
			itr++;
		} while(count>itr);
		
		return glErr/count;
	}
	//update the weights for current iteration
	static void updateWeights(double err, double[] weights, double[] currFeates){

		for(int i=0; i < featNumber; i++)
			weights[i] +=  err * currFeates[i];
		weights[featNumber] += err;
	}
	//compute the prediction based on inputs
	static int computePrediction(double[] weights, double[] currFeates){

		double sum = 0;

		for(int i = 0; i < featNumber; i++)
			sum += weights[i]*currFeates[i]*LEARNING_RATE;
		sum += weights[featNumber];

		return (sum >= threshold) ? 1 : 0;
	}

	//parse the output value into integer
	static double parseOutput(String currLine, String realOutputs){

		String[] splitOutputs = realOutputs.split(" ");
		String[] splitCurr = currLine.split(" ");

		if(splitCurr[1].trim().equals(splitOutputs[0].trim()))
			return 1;
		else
			return 0;
	}

	//parse the input line into features
	static double[] parseFeatures(String line, String[] currFeats){

		double[] returned = new double[featNumber];
		String[] splitted = line.split(" ");

		for(int i = featNumber-1; i>=1; i--){
			if(splitted[i].trim().equals(currFeats[i].split(" ")[0].trim()))
				returned[i] = 1;
			else
				returned[i] = 0;
		}

		return returned;
	}

	//Find the item count for each file
	static int findItemCount(Scanner file){

		int returned = -1;
		String currLine = null;

		while(file.hasNext() && returned==-1){

			currLine = file.nextLine();

			if(currLine.equals("") || currLine.startsWith("//"))
				continue;
			else
				returned = Integer.parseInt(currLine.trim());
		}

		return returned;

	}

	//Find the two output values, separate them by a space
	static String findOutputs(Scanner file){

		String returned = null;
		String currLine = null;

		while(file.hasNext() && returned == null){

			currLine = file.nextLine();

			if(currLine.equals("") || currLine.startsWith("//"))
				continue;
			else
				returned = currLine.trim() + " " + file.nextLine().trim(); 
		}

		return returned;

	}

	//find the values of each feature.
	//The first feature will be encoded as 0, second as 1
	static String[]  findFeatsVals(Scanner file){

		String[] returned = new String[featNumber];
		int count = 0;
		String currLine = null;

		while(file.hasNext() && count <= featNumber-1){

			currLine = file.nextLine();

			if(currLine.equals("") || currLine.startsWith("//"))
				continue;
			else {
				String[] splitted = currLine.split("-");
				returned[count] = splitted[splitted.length-1].trim();
				count++;
			}

		}
		return returned;
	}

	//find the number of features in a file
	static int findFeatsNumber(Scanner file){

		int returned = -1;
		String currLine = null;

		while(file.hasNext() && returned == -1){

			currLine = file.nextLine();

			if(currLine.equals("") || currLine.startsWith("//"))
				continue;
			else 
				returned = Integer.parseInt(currLine.trim());
		}

		return returned;
	}

}
