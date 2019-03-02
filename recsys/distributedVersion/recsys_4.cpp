#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <random>
#include "boost/tokenizer.hpp"
#include "core/engine.hpp"
#include "lib/ml/data_loader.hpp"
#include "lib/ml/feature_label.hpp"
#include "load_rating.hpp"

using namespace std;
using namespace husky;

typedef Eigen::MatrixXd MatrixT;


base::BinStream& operator<<(base::BinStream& stream, const MatrixT& matrix) {
    stream << size_t(matrix.rows());
    stream << size_t(matrix.cols());
    for (int i = 0; i < matrix.rows(); i++)
        for (int j = 0; j < matrix.cols(); j++)
            stream << i << j << (double) matrix(i, j);
}

base::BinStream& operator>>(base::BinStream& stream, MatrixT& matrix) {
    size_t row_length, col_length;
    stream >> row_length >> col_length;
    matrix = MatrixT(row_length, col_length);
    double value;
    for (int i = 0; i < row_length; i++) {
        for (int j = 0; j < col_length; j++) {
            stream >> i >> j >> value;
            matrix.coeffRef(i, j) = value;
        }
    }
    return stream;
}

// int dim = 4;
//default feature dimension : 4
class FeatureT {
    public:
    using KeyT = int;
    FeatureT() = default;
    explicit FeatureT(const KeyT& k) : id_num(k) {
        // static default_random_engine e(time(0));
        // static uniform_real_distribution<double> n(-1,1);
        // feature = MatrixT::Zero(dim,1).unaryExpr([](double dummy){return n(e);});
    }
    const KeyT& id() const { return id_num; }

    // Serialization and deserialization
    friend BinStream& operator<<(BinStream& stream, const FeatureT& u) {
        stream << u.id_num << u.feature;
        return stream;
    }
    friend BinStream& operator>>(BinStream& stream, FeatureT& u) {
        stream >> u.id_num >> u.feature;
        return stream;
    }

    KeyT id_num;
    MatrixT feature;
    vector<pair<int,float>> records;
};

class recsys{
private:
    int dim;
    // get model config parameters
    double lambda_;
    int epochnum;

    ObjList<rating_obj>& train_set = ObjListStore::create_objlist<rating_obj>();
    ObjList<rating_obj>& test_set = ObjListStore::create_objlist<rating_obj>();

    //define feature list
    ObjList<FeatureT>& user_list = ObjListStore::create_objlist<FeatureT>();
    ObjList<FeatureT>& item_list = ObjListStore::create_objlist<FeatureT>();

    int num_samples;
    int num_test;
    int num_users;
    int num_items;

public:
    recsys(){
        dim = stoi(Context::get_param("dimension"));
        lambda_ = stod(Context::get_param("lambda_"));
        epochnum = stoi(Context::get_param("epochnum"));

        // load data
        load_rating(Context::get_param("train"), train_set);
        load_rating(Context::get_param("test"), test_set);

        globalize(train_set);
        globalize(test_set);

        globalize(user_list);
        globalize(item_list);

        auto& init_user = ChannelStore::create_push_channel<pair<int,float>>(train_set, user_list);
        auto& init_item = ChannelStore::create_push_channel<pair<int,float>>(train_set, item_list);

        //first . initialization
        list_execute(train_set, [this,&init_item,&init_user](rating_obj& this_obj) {
            init_user.push(make_pair(this_obj.item_id , this_obj.rating) , this_obj.user_id);
            init_item.push(make_pair(this_obj.user_id , this_obj.rating) , this_obj.item_id);
        });

        list_execute(item_list, [&init_item,this](FeatureT& f_obj) {
            auto t = (unsigned int)clock();
            srand(t+f_obj.id());
            f_obj.feature = MatrixT::Random(dim , 1);
            f_obj.records = init_item.get(f_obj);
        });
        list_execute(user_list, [&init_user,this](FeatureT& f_obj) {
            auto t = (unsigned int)clock();
            srand(t+f_obj.id());
            f_obj.feature = MatrixT::Random(dim , 1);
            f_obj.records = init_user.get(f_obj);
        });
    }

    void print_info(){
        // get the number of global records
        lib::Aggregator<int> num_samples_agg(0, [](int& a, const int& b) { a += b; });
        num_samples_agg.update(train_set.get_size());

        lib::Aggregator<int> num_test_agg(0, [](int& a, const int& b) { a += b; });
        num_test_agg.update(test_set.get_size());

        //get the number of unique user
        lib::Aggregator<int> num_user_agg(0, [](int& a, const int& b) { a += b; });
        num_user_agg.update(user_list.get_size());

        //get the number of unique item
        lib::Aggregator<int> num_item_agg(0, [](int& a, const int& b) { a += b; });
        num_item_agg.update(item_list.get_size());

        lib::AggregatorFactory::sync();

        num_samples = num_samples_agg.get_value();
        num_test = num_test_agg.get_value();
        num_users = num_user_agg.get_value();
        num_items = num_item_agg.get_value();
        if (Context::get_global_tid() == 0) {
            LOG_I << "lambda =  " << lambda_;
            LOG_I << "iteration number = " << epochnum;
            LOG_I << "Training set size = " << num_samples;
            LOG_I << "Test set size = " << num_test;
            LOG_I << "user size = " << num_users;
            LOG_I << "item size = " << num_items;
        }
    }

    double compute_rmse(auto& dataset , int& num){
        lib::Aggregator<double> error_agg(0.0, [](double& a, const double& b) { a += b; });
        // error_agg.to_reset_each_iter();
        auto& broad_channel = ChannelStore::create_broadcast_channel<int, MatrixT>(user_list);

        //calculate  loss
        list_execute(user_list,{},{&broad_channel}, [&broad_channel](FeatureT& f_obj) {
            broad_channel.broadcast(f_obj.id() , f_obj.feature);
        });

        list_execute(item_list,{&broad_channel},{},[&broad_channel,&error_agg,this](FeatureT& f_obj) {
            vector<pair<int,float>>::iterator iter;
            for(iter = f_obj.records.begin();iter!=f_obj.records.end();iter++){
                double predict = (broad_channel.get(iter->first).transpose() * f_obj.feature)(0,0);
                double err = iter->second-predict;
                error_agg.update(pow(err,2));
            }
        });
        lib::AggregatorFactory::sync();
        return sqrt(error_agg.get_value()/num);
    }

    void train(){
        //channel communication

        auto& user_broad = ChannelStore::create_broadcast_channel<int, MatrixT>(user_list);
        auto& item_broad = ChannelStore::create_broadcast_channel<int, MatrixT>(item_list);


        list_execute(item_list,{},{&item_broad},  [&item_broad](FeatureT& f_obj) {
            item_broad.broadcast(f_obj.id(),f_obj.feature);
        });

        // Main loop
        auto start = std::chrono::steady_clock::now();
        for (int k = 0; k < epochnum; k++) {
            list_execute(user_list,{&item_broad},{&user_broad}, [&item_broad,&user_broad,this](FeatureT& f_obj) {
                vector<pair<int,float>>::iterator iter;
                MatrixT u_divider = MatrixT::Zero(dim,dim);
                MatrixT u_dividend = MatrixT::Zero(dim,1);

                for(iter = f_obj.records.begin();iter!=f_obj.records.end();iter++){
                    auto i_feature = item_broad.get(iter->first);
                    u_divider += i_feature * i_feature.transpose();
                    u_dividend += i_feature * iter->second;
                }
                u_divider = u_divider + lambda_ * MatrixT::Identity(dim,dim);
                f_obj.feature = u_divider.inverse() * u_dividend;
                user_broad.broadcast(f_obj.id() , f_obj.feature);
            });

            list_execute(item_list,{&user_broad},{&item_broad},[&user_broad,&item_broad,&k,this](FeatureT& f_obj) {
                vector<pair<int,float>>::iterator iter;
                MatrixT i_divider = MatrixT::Zero(dim,dim);
                MatrixT i_dividend = MatrixT::Zero(dim,1);
                for(iter = f_obj.records.begin();iter!=f_obj.records.end();iter++){
                    auto u_feature = user_broad.get(iter->first);
                    i_divider += u_feature * u_feature.transpose();
                    i_dividend += u_feature * iter->second;
                }
                i_divider = i_divider + lambda_ * MatrixT::Identity(dim,dim);
                f_obj.feature = i_divider.inverse() * i_dividend;

                if(k!=epochnum-1){
                    item_broad.broadcast(f_obj.id() , f_obj.feature);
                }
            });

            double train_rmse = compute_rmse(train_set , num_samples);
            double test_rmse = compute_rmse(test_set , num_test);
            if (Context::get_global_tid() == 0) {
                LOG_I << k << " epoch :";
                LOG_I << "loss on train set: " << train_rmse;
                LOG_I << "loss on test set: " << test_rmse;
            }
        }

        auto end = std::chrono::steady_clock::now();
        // Show result
        if (Context::get_global_tid() == 0) {
            LOG_I << "Time per epoch: "
                         << std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count() / epochnum;
        }
    }

    void predict(){

    }


};

void rec_task(){
    recsys rec;
    rec.print_info();
    rec.train();
}


int main(int argc, char** argv) {
    vector<string> args(
        {"hdfs_namenode", "hdfs_namenode_port", "train", "test", "epochnum", "lambda_"});
    if (init_with_args(argc, argv, args)) {
        run_job(rec_task);
        return 0;
    }
    return 1;
}
