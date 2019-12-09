using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace HelloWorld
{
    class Program
    {
        public class HouseData
        {
            public float Size { set; get; }
            public float Price { set; get; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { set; get; }
        }

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // 1. Prepare Data
            HouseData[] houseData =
            {
                new HouseData{Size=1.1f, Price=1.2f},
                new HouseData{Size=1.9f, Price=2.3f},
                new HouseData{Size=2.8f, Price=3.0f},
                new HouseData{Size=3.4f, Price=3.7f},
            };
            IDataView trainingData = mlContext.Data.LoadFromEnumerable(houseData);

            // Define Model
            var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

            // Training Model
            var model = pipeline.Fit(trainingData);

            // Predict
            var size = new HouseData { Size = 2.5f };
            var price = mlContext.Model.CreatePredictionEngine<HouseData, Prediction>(model).Predict(size);

            Console.WriteLine($"Predicted price for size: {size.Size * 1000} sq ft = {price.Price * 100:C}k");

            //
            HouseData[] testHouseData =
            {
                new HouseData { Size = 1.1f, Price = 0.98f},
                new HouseData { Size = 1.9f, Price = 2.1f},
                new HouseData { Size = 2.8f, Price = 2.9f},
                new HouseData { Size = 3.4f, Price = 3.6f},
            };

            var testHouseDataView = mlContext.Data.LoadFromEnumerable(testHouseData);
            var debug = testHouseDataView.Preview();
            var testPriceDataView = model.Transform(testHouseDataView);

            var metrics = mlContext.Regression.Evaluate(testPriceDataView, labelColumnName: "Price");

            Console.WriteLine($"R^2: {metrics.RSquared:0.##}");
            Console.WriteLine($"RMS Error: {metrics.RootMeanSquaredError:0.##}");
        }
    }
}
