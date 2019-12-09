using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    class Program
    {
        private static string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        static readonly string _dataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "naver_labelled.txt");

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Load Data
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // BuildAndTrainModel
            var estimator = mlContext.Transforms.Text
                .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features", maximumNumberOfIterations: 100));

            Console.WriteLine("=============== Create and Train the Model ===============");
            var model = estimator.Fit(splitDataView.TrainSet);
            Console.WriteLine("=============== End of training ===============");
            Console.WriteLine();

            // Evaluate            
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
            IDataView predictions = model.Transform(splitDataView.TestSet);

            CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            // UseModelWithSingleItem
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "이 영화 정말 재미없어요"
            };

            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset===============");
            Console.WriteLine();
            Console.WriteLine($"Sentiment: {resultPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {resultPrediction.Probability} ");
            Console.WriteLine("=============== End of Predictions ===============");
            Console.WriteLine();

            // UseModelWithBatchItems
            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData { SentimentText = "지루한 영화에요" },
                new SentimentData { SentimentText = "이거 정말 최고에요!" },
                new SentimentData { SentimentText = "올해의 영화로 손꼽고 싶군요" }
            };

            IDataView batchComments = mlContext.Data.LoadFromEnumerable(sentiments);
            predictions = model.Transform(batchComments);

            // Use model to predict whether comment data is Positive (1) or Negative (0).
            IEnumerable<SentimentPrediction> predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: false);

            Console.WriteLine();
            Console.WriteLine("=============== Prediction Test of loaded model with multiple samples ===============");
            foreach (SentimentPrediction prediction in predictedResults)
            {
                Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction:{ (Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability:{ prediction.Probability}");
            }
            Console.WriteLine("=============== End of predictions ===============");
        }
    }
}
