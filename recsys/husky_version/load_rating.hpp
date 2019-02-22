#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include "boost/tokenizer.hpp"
#include "base/assert.hpp"
#include "core/executor.hpp"
#include "core/objlist.hpp"
#include "io/input/inputformat_store.hpp"
#include "lib/aggregator_factory.hpp"




class rating_obj {
    public:
    using KeyT = int;

    rating_obj() = default;
    explicit rating_obj(const KeyT& i , const int& a, const int& b , const float& c) : index(i) {
        //id = i;
        user_id = a;
        item_id = b;
        rating = c;
    }
    const KeyT& id() const { return index; }

    friend husky::BinStream& operator<<(husky::BinStream& stream, const rating_obj& r) {
        stream << r.index << r.user_id << r.item_id << r.rating;
        return stream;
    }

    friend husky::BinStream& operator>>(husky::BinStream& stream, rating_obj& r) {
        stream >> r.index >> r.user_id >> r.item_id >> r.rating;
        return stream;
    }

    KeyT index;
    int user_id;
    int item_id;
    float rating;

};



void load_rating(std::string url, husky::ObjList<rating_obj>& data) {

    //husky::lib::Aggregator<int> num_sample(0, [](int& a, const int& b) { a += b ; });
    int sum = 0 ;

    boost::char_separator<char> sep("\t");
    boost::tokenizer<boost::char_separator<char>> tok(url, sep);
    for (auto& w : tok) {

        //husky::LOG_I << w;

        auto& infmt = husky::io::InputFormatStore::create_line_inputformat();
        infmt.set_input(w);

        auto& ac = husky::lib::AggregatorFactory::get_channel();

        std::function<void(boost::string_ref)> parser;
        parser = [&](boost::string_ref chunk) {
            if (chunk.empty())
                return;
            boost::char_separator<char> sep("\t");
            boost::tokenizer<boost::char_separator<char>> tok1(chunk, sep);
            int k = 0;
            std::vector<std::string> str(3);
            for (auto& s : tok1) {
                if (k < 3 ){
                    str[k] = s;
                }
                else{}
                k++;
            }
            sum++ ;
            rating_obj this_obj(sum ,std::stoi(str[0]) , std::stoi(str[1]) , std::stof(str[2]));  // create a data object
            data.add_object(this_obj);
        };

        husky::load(infmt, {&ac}, parser);
    }

}
