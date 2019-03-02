import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import scala.math.sqrt

// mainly use Rating, MatrixFactorizationModel, ALS

object RecommendationExample {
    def main(args: Array[String]){
        val conf = new SparkConf().setAppName("CollaborativeFilteringExample")
        val sc = new SparkContext(conf)

        // Load and parse the training data
        val train_data = sc.textFile(args(0))
        val test_data = sc.textFile(args(1))
        val rank = 4
        val numIterations = args(2).toInt
        val train_rating = train_data.map(_.split('\t') match { case Array(user, item, rate) =>
          Rating(user.toInt, item.toInt, rate.toDouble)
        })

        // Build the recommendation model using ALS
        val time1=System.currentTimeMillis()
        val model = ALS.train(train_rating, rank, numIterations, 0.01s
        val time2=System.currentTimeMillis()
        val epochtime = (time2 - time1)/(1000.0 * numIterations)
        println(s"every epoch time = $epochtime")

        // Evaluate the model on test data
        val test_rating = test_data.map(_.split('\t') match { case Array(user, item, rate) =>
          Rating(user.toInt, item.toInt, rate.toDouble)
        })
        val usersProducts = test_rating.map { case Rating(user, product, rate) =>
          (user, product)
        }
        val predictions =
          model.predict(usersProducts).map { case Rating(user, product, rate) =>
            ((user, product), rate)
          }
        val ratesAndPreds = test_rating.map { case Rating(user, product, rate) =>
          ((user, product), rate)
        }.join(predictions)
        val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
          val err = (r1 - r2)
          err * err
        }.mean()
        val rmse = sqrt(MSE)
        println(s"Square Mean Squared Error = $rmse")

        // Save and load model
        // model.save(sc, "target/tmp/myCollaborativeFilter")
        // val sameModel = MatrixFactorizationModel.load(sc, "target/tmp/myCollaborativeFilter")

        sc.stop()
    }

}
