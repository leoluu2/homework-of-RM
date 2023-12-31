//目前这个代码不会报错，但是有错误，他推理有问题，一直输出一个值
//推测可能是模型导出时的问题，也可能是这个代码的问题

//别忘跑之前roscore！！！！！

#include <ros/ros.h>
#include <cstddef>
// #include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>

// #include <cpu_execution_provider.h>

ros::Publisher classification_pub;
size_t num_input_nodes,num_output_nodes;
// std::vector<const char*> input_node_names,output_node_names;
std::vector<const char*> input_node_names = {"actual_input"};
std::vector<const char*> output_node_names = {"actual_output"};
std::vector<Ort::Value> ort_inputs;
int num_classes = 10;

// // 初始化ONNX Runtime环境
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime");
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // //创建ONNX模型会话，后面可以使用该会话进行推理
    //注意路径，这里是相对于catkin_ws的位置，否则会failed to download model
    Ort::Session onnx_session(env, "./src/detect_pkg/src/mnist_model.onnx", session_options);
    Ort::AllocatorWithDefaultOptions allocator;//打印模型的输入层：node_name,types,shape,stc
std::vector<int64_t> input_node_dims = {1,1,28, 28};
size_t input_tensor_size = 1*1*28*28;//这里要和dims统一



// softmax 函数的实现
void softmax(float* data, int size) {
    // 找到最大值（用于数值稳定性）
    float max_val = *std::max_element(data, data + size);

    // 计算指数并求和
    float sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum_exp += data[i];
    }

    // 归一化
    for (int i = 0; i < size; ++i) {
        data[i] /= sum_exp;
    }
}


// 注意这里是imageConstPtr 而不是ImagePtr！！！！！，否则报错到崩溃
//这里只是用到msg，不会更改，最多对副本进行更改！！！！
void detect_callback(const sensor_msgs::ImageConstPtr& msg){//别傻乎乎传值

    //MNIST 是24*24黑白图像，但是相机拍的是彩色的，所以要用opencv进行转换
    //在ros中，sensor_msgs::Image消息默认使用BGR8编码
    //因为参数是imageconstptr，所以拷贝副本
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    cv::Mat gray_img;
    cv::cvtColor(cv_ptr->image, gray_img, cv::COLOR_BGR2GRAY);
    
    // 缩放图像到24x24
    cv::Mat resized_img;
    cv::resize(gray_img, resized_img, cv::Size(24, 24));
    // printf("连到detect——call回调函数了\n");测试
    // cv::imshow("img",resized_img);测试

    //-----------------以上部分得到待推理图像的cv::Mat形式------------------

    // 每次循环中都重新加载ONNX模型和创建ONNX会话。这样的做法会导致每次推理都需要重新加载模型
    // 这不仅效率低下，而且可能会导致每次推理结果都相同的问题

    //添加模型推理

    // // // 初始化ONNX Runtime环境
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime");
    // Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // // //创建ONNX模型会话，后面可以使用该会话进行推理
    // //注意路径，这里是相对于catkin_ws的位置，否则会failed to download model
    // Ort::Session onnx_session(env, "./src/detect_pkg/src/mnist_model.onnx", session_options);
    // Ort::AllocatorWithDefaultOptions allocator;//打印模型的输入层：node_name,types,shape,stc

    //  // print number of model input nodes
    // size_t num_input_nodes = onnx_session.GetInputCount();
    // // char* input_name = onnx_session.GetInputName(0, allocator);
    // // std::vector<const char*> input_node_names = {input_name};
    // // Ort::GetApi().ReleaseValue(input_name);//严谨
    // std::vector<const char*> input_node_names = {"actual_input"};
    // std::vector<const char*> output_node_names = {"actual_output"}; 

    // int num_classes = 10;
    // size_t input_tensor_size = 1*1*28*28;//这里要和dims统一
    // //// 创建输入tensor
    // // ONNX模型的输入要求，需要创建适当形状的输入Tensor，并将图像数据填充到其中
    // // 这通常涉及使用ONNX Runtime提供的API来创建和填充Tensor
    // // Ort::MemoryInfo 允许你描述将要创建的张量的内存信息，包括分配器和内存类型
    // // Ort::Value 的创建需要提供关于张量的信息，包括内存信息、数据类型、形状和大小
    // // dims - batch_size,channels,height,width
    // std::vector<int64_t> input_node_dims = {1,1,28, 28};
    // // size_t input_tensor_size = 10 * 20; 
    // std::vector<float> input_tensor_values(input_tensor_size);
    // for (unsigned int i = 0; i < input_tensor_size; i++)
    //     input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    // // create input tensor object from data values
    // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
    //      input_tensor_size, input_node_dims.data(), 4);//这里的4是指input_node_dims的维度
    // assert(input_tensor.IsTensor());

    // std::vector<Ort::Value> ort_inputs;
    // ort_inputs.push_back(std::move(input_tensor));
    // // ort_inputs.push_back(std::move(input_mask_tensor));
    // // score model & input tensor, get back output tensor
    //---------------------------------------------------------------------

    // 更新 ort_inputs 向量内容
    // std::vector<float> input_tensor_values(resized_img.data, resized_img.data + resized_img.total());
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    //     allocator, input_tensor_values.data(), input_tensor_values.size(),
    //     input_node_dims.data(), input_node_dims.size());
    // ort_inputs = {std::move(input_tensor)};

    auto output_tensors = onnx_session.Run(
            Ort::RunOptions{nullptr}, input_node_names.data(), 
            ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 1);
    // auto output_tensors = onnx_session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), nullptr, 0);

    
    // Get pointer to output tensor float values
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    // float* floatarr_mask = output_tensors[1].GetTensorMutableData<float>();
        // printf("Done!\n");
        // printf("%lf\n",*floatarr);

    // 对模型输出进行 softmax 处理
    softmax(output_data, num_classes);

    // 假设输出是概率分布，找到最大概率的索引即为分类结果
    float max_probability = output_data[0];
    int max_index = 0;

    for (int i = 1; i < num_classes; ++i) {
        if (output_data[i] > max_probability) {
            max_probability = output_data[i];
            max_index = i;
        }
    }
    printf("%d\n",max_index);

    // // 发布分类信息
    // std_msgs::Int32 classification_msg;
    // classification_msg.data = max_index;
    // classification_pub.publish(classification_msg);
}

/*
void detect_callback(const sensor_msgs::ImageConstPtr& msg){
    // cv_bridge::CvImagePtr cv_ptr;
    // try {
    //     cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    // } catch (cv_bridge::Exception& e) {
    //     ROS_ERROR("cv_bridge exception: %s", e.what());
    //     return;
    // }
}*/
//只有初始化之后的节点才能和ros产生连接
int main(int argc, char *argv[])
{
    
    ros::init(argc,argv,"detect_node");

    //node的handle变量的是节点和ros通讯的关键
    ros::NodeHandle nh;//ros相当于大豪宅，nodehandle就是里面的管家
    //pubulisher 是 发送消息的工具，类比为能够发送QQ消息的收集
    //advertise的第二个参数是能排队的消息的坑位数
    //话题名不能是中文，不能有空格
    // ros::Publisher pub = nh.advertise<std_msgs::String>("QUICK_SHANGCHE",10);
     
    // // // 初始化ONNX Runtime环境
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime");
    // Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // // //创建ONNX模型会话，后面可以使用该会话进行推理
    // //注意路径，这里是相对于catkin_ws的位置，否则会failed to download model
    // Ort::Session onnx_session(env, "./src/detect_pkg/src/mnist_model.onnx", session_options);
    // Ort::AllocatorWithDefaultOptions allocator;//打印模型的输入层：node_name,types,shape,stc
    //// num_input_nodes = onnx_session.GetInputCount();
    //// num_output_nodes = onnx_session.GetOutputCount();
    // //输入输出名，输入维度
    // std::vector<const char*> input_node_names(num_input_nodes);
    // std::vector<const char*> output_node_names(num_output_nodes);
    // std::vector<int64_t> input_node_dims;

        //添加模型推理

    // // // 初始化ONNX Runtime环境
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime");
    // Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // // //创建ONNX模型会话，后面可以使用该会话进行推理
    // //注意路径，这里是相对于catkin_ws的位置，否则会failed to download model
    // Ort::Session onnx_session(env, "./src/detect_pkg/src/mnist_model.onnx", session_options);
    // Ort::AllocatorWithDefaultOptions allocator;//打印模型的输入层：node_name,types,shape,stc

     // print number of model input nodes
    size_t num_input_nodes = onnx_session.GetInputCount();
    // char* input_name = onnx_session.GetInputName(0, allocator);
    // std::vector<const char*> input_node_names = {input_name};
    // Ort::GetApi().ReleaseValue(input_name);//严谨
    // input_node_names = {"actual_input"};
    // output_node_names = {"actual_output"}; 

   
    // size_t input_tensor_size = 1*1*28*28;//这里要和dims统一
    //// 创建输入tensor
    // ONNX模型的输入要求，需要创建适当形状的输入Tensor，并将图像数据填充到其中
    // 这通常涉及使用ONNX Runtime提供的API来创建和填充Tensor
    // Ort::MemoryInfo 允许你描述将要创建的张量的内存信息，包括分配器和内存类型
    // Ort::Value 的创建需要提供关于张量的信息，包括内存信息、数据类型、形状和大小
    // dims - batch_size,channels,height,width
    
    // size_t input_tensor_size = 10 * 20; 


    std::vector<float> input_tensor_values(input_tensor_size);
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);
    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
         input_tensor_size, input_node_dims.data(), 4);//这里的4是指input_node_dims的维度
    assert(input_tensor.IsTensor());

    // // std::vector<Ort::Value> ort_inputs;//这个需要放在这里
    ort_inputs.clear();
    ort_inputs.push_back(std::move(input_tensor));


    //usb_cam默认使用"/usb_cam/image_raw"作为话题名称
    //// ros::Subscriber sub = nh.subscribe("/usb_cam/image_raw",10,detect_callback);
    //通常使用image_transport订阅图形话题
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/usb_cam/image_raw",10,detect_callback);
    printf("完成回调\n");


    // 设置分类信息发布器
    // classification_pub = nh.advertise<std_msgs::Int32>("mnist_classification", 10);

    // ros::Rate loop_rate(10);//1s发10条

    //可以用ros::spin()代替
    while(ros::ok()){//这样可以用ctrl c停止，但是while(true)不可以

        ros::spinOnce();  // 处理一次ROS回调!!!!!
        // loop_rate.sleep();
    }
    return 0;
}
