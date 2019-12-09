using System;
using System.IO;
using Microsoft.ML;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "taeyo_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "taeyo_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0);

            // Prepare Data
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

            // Prepare Pipeline
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext)
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features", maximumNumberOfIterations: 1000))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Training Model
            _trainedModel = pipeline.Fit(_trainingDataView);

            // Single Prediction
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };

            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area}===============");

            // Evaluate
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"* Metrics for Multi-class Classification model - Test Data ");
            Console.WriteLine($"*----------------------------------------------------------------------------------------------------------- - ");
            Console.WriteLine($"* MicroAccuracy: {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"* MacroAccuracy: {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"* LogLoss: {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"* LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            // Save Model
            _mlContext.Model.Save(_trainedModel, _trainingDataView.Schema, _modelPath);

            // Predict Issue
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue()
            {
                //Title = "socket 중계 서버..",
                //Description = "오랜만에 질문 올립니다.  tcp socket 중계 서버를 만들어야 하는데요..  간단하게 얘기하면 n개의 장비가 있고 n개의 컨트롤 시스템이 존재 각 장비는 지정 컨트롤 시스템과 정보 송/수신. 말하자면 1:1의 양방향 통신이 n개가 관리 가능해야 한다는 얘긴데 .  socket.poll, threadPool.. 등등... 뭘 어찌 구현할지 좀 감이 안오네요. 비슷한 구현 해보신 분 힌트좀 주세요..."
                Title = "오랜만에 송년회 어떠세요?",
                Description = "참석하고 싶은 마음이 불처럼 타오르는 분들 계신가요!!!!"
            };

            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
            var prediction2 = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {prediction2.Area} ===============");
        }
    }
}
