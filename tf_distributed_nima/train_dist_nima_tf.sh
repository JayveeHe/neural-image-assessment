source /opt/meituan/hadoop-gpu/bin/hadoop_user_login_centos7.sh hadoop-dpsr
source /opt/meituan/tensorflow-release/local_env.sh
export JAVA_HOME="/usr/local/java"
unset CLASSPATH

${AFO_TF_HOME}/bin/tensorflow-submit.sh -conf tf_disttrain_conf.xml -files nima.py,w2v_utils.py,data_utils.py,dist_train.py,network_builder.py,tfrecord_builder.py