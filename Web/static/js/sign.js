// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function () {
    // 获取登录按钮
    var loginBtn = document.getElementById('loginBtn');

    // 添加点击事件监听器
    loginBtn.addEventListener('click', function () {
        // 弹出提示窗口
        var userResponse = window.confirm('此功能尚未开发！点击确定返回。');
        
        // 根据用户点击操作
        if (userResponse) {
            // 确定后关闭窗口或进行其他操作
            console.log('用户点击了确定');
        } else {
            // 用户选择取消可以执行其他逻辑
            console.log('用户点击了取消');
        }
    });
});
