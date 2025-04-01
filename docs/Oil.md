# 听课笔记

## Axios网络请求

1. POST请求的时候要用json格式做特殊的处理。
   -  ![image-20250308103342056](images/image-20250308103342056.png)
2. 本来跨域是不允许cookie共享的。但是可以通过Access-Control-Allow-Credentials: true设置访问cookie
   1. allowOrigins里面写*会报错，可以替换成allowedOriginPatterns("*")

3. 回调函数中的作用域会发生变化的。
   1. 解决方案，弄一个箭头函数，让其作用域继承父集
   2. ![image-20250308110448171](images/image-20250308110448171.png)

## Vuex

要清楚这一小节究竟在干嘛。

因为组件套多了，数据往下个节点传代码会很冗杂，所以引入这个会方便很多。所以要学习Vuex。

因此，如果你的项目中，并没有很多的传导，那么是不需要这一届的内容的。

<font color="yellow">对于本小节：</font>

state保存数据，mutations更改数据，Getters过滤数据，Action异步处理，然后每种方法的具体使用根据文档来就好了

要理解，**store相当于一个全局变量**，这边全局变量的数据会和视图绑定，当我们修改这个变量的值时，我们应该遵照他的规则，提供一个方法用于修改数据，而不是直接修改数据。

![image-20250309145415013](images/image-20250309145415013.png)

那么之后的所有内容均是在按照他的要求，学习如何进行修改。

- Mutation一般做一些同步的操作，可以在该函数中直接修改状态
  - 在这里做异步的操作，是没法记录的。
- Action一般做一个异步的操作
  - 不可以直接修改状态，只能通过提交mutation来修改。这样做的目的是为了可以在Action中记录state的变化
  - 

## mockjs

相当于后端的postman，用来模拟服务器的相应的。用mock进行拦截前端的请求，然后将设置后的res返回。

### 注销

只需要将Cookies中的token给清空即可。



# 项目一：vue-admin-template模板快速使用入门,

1. 网址：https://www.bilibili.com/video/BV1gr421F7ok/?p=52&vd_source=cb88355d0fa4b837438e420a28b4f3b2

   - 做到了P34,登出功能完成。

   - **已经完成**了，这个框架大致的了解，已经如何获取该框架中前端的发送、后端接受。已经登录、获取token、登出。之后是进行角色

   - **未完成**：用户角色分类、侧边栏的动态路由。

2. 资源名
   1. IDEA项目名：ukoko-list
   2. ![image-20250311110537417](images/image-20250311110537417.png)
   3. 数据库为：
      - vue
   4. 测试：ukoko测试
   5. 前端项目：D:\IdeaProjects\SpringBootLearing\vue-admin-template-master> 

# 项目二：学会springboot+vue的简单的增删改查

网址：https://www.bilibili.com/video/BV12p4y167ny?spm_id_from=333.788.videopod.episodes&vd_source=cb88355d0fa4b837438e420a28b4f3b2&p=2

- 项目地址：直接在ukoko-list创建一个新的module

### 1.相关注解的理解

- Service中可以不加@Service，在具体的Impl中加入@Service和@Transactional 

- @Autired和@Resource类似，但是具体实现和使用有区别。具体区别如下：
  - 
- @Mapper和@Repository类似，但也有区别，具体区别如下：
  - https://blog.csdn.net/qq_44421399/article/details/109825479	
- @RequestBody、@RequestParam 和 @Param 的区别与用法:
  - https://blog.csdn.net/qq_62851576/article/details/142973756
- 

### 2.遇到的问题总结

- 1 出现**org.apache.ibatis.binding.BindingException: Parameter ‘username’ not found. Available parameters are [0, 1, param1, param2]** ，经过测试，发现是Controller、Service都可以进入，那么是Mapper的问题。

  - 出现场景：在Controller中加入了@RequestBody的注解，因为要传递username和password。但是在Mapper中，<font color = 'yellow'>除了在Class层中加入了@Mapper，在方法中没有加入任何注解。</font>结果发现是因为<font color = 'yellow'>平时用注解的方式是不需要加入@Param的注解的，而xml的方式中最好是加入，防止出现各种别的问题。</font>在下列的博客中详细说明了该问题：https://blog.csdn.net/m0_59960942/article/details/136395223
  - 总结就是：<font color = 'pink'>注解的方式，可以不再mapper.class中加入注释。但是如果是用xml的方式，则最好在Mapper中加入@Param的注解。不影响Controller加入@RequestBody</font>

  

- 2 当在Vue的控件中，加入 : 时，是`v-bind`指令的简写形式。Vue使用指令来操作或响应DOM的行为。<font color = 'orange'>`v-bind`用于动态地绑定HTML属性、</font>组件prop等值到Vue实例的数据上。

  - ```
    <img :src="imageSrc">
    ```

  - 这里的`:src`是`v-bind:src`的缩写，表示将`img`标签的`src`属性绑定到Vue实例中的`imageSrc`数据属性。这意味着当Vue实例中的`imageSrc`值发生变化时，`img`元素的`src`属性也会自动更新为新的值。

  

- **3 v-model和v-bind的区别**

  - v-model主要用于表单元素（如`<input>`、`<textarea>`、`<select>`等）与组件状态之间的双向数据绑定。它简化了输入框和其他表单控件的数据同步过程。

    - ```
      <el-input v-model="queryInfo.query" placeholder="请输入搜索内容"></el-input>
      ```

  - v-bind用于父组件向子组件传递数据或配置静态/动态属性

    - ```
      <img :src="imageSrc" alt="描述">
      <!-- 或者 -->
      <custom-component :user-info="userInfo"></custom-component>
      ```

      


- **3 分页用的方式是每次重新sql查询。我觉得太麻烦了。课程的代码如下：**

  - ```
    methods:{
            //获取所有用户信息
            async  getUserList() {
               //发送get请求
               const {data:res}  = await this.$http.get("allUser",{params:this.queryInfo})
               this.userList = res.data; //都是后端查询到的结果分装到hashmap中的。
               this.total = res.numbers;
            },
            //监听pageSize改变的事件
            handleSizeChange(newSize){
                this.queryInfo.pageSize = newSize;
                this.getUserList(); // 数据发生改变重新申请数据
            },
            // 监听pageNum改变的事件
            handleCurrentChange(newPage){
                this.queryInfo.pageNum = newPage;
                this.getUserList(); // 数据发生改变重新申请数据
            },
    ```

  - 应该有方法改进的，看看。

- **4 `.then()` 和 `await` 都是用来处理异步操作的，但它们的使用方式和适用场景有所不同。**

  - **区别**：`.then()` 更加底层，适合用于构建Promise链；而 `await` 提供了更加直观、简洁的语法糖，使得异步代码更容易理解和维护。此外，`await` 只能在 `async` 函数内使用，而 `.then()` 没有这个限制。
  - **联系**：两者都是用来处理异步操作的工具，并且都可以结合 Promise 使用。实际上，`await` 在底层也是通过 Promise 实现的。

- **5.@Param和@RequestParam的区别与联系**

  - `@Param` 和 `@RequestParam` 注解在功能和使用场景上有明显的区别，它们分别用于不同的层和服务目的。

  - `@Param`

    - **所属框架**：MyBatis（或其他类似ORM框架）

    - **作用位置**：DAO层或Mapper接口中

    - **用途**：`@Param` 注解主要用于给方法参数命名，以便在SQL映射文件中引用这些参数。它使得在编写动态SQL时能够更清晰地引用传入的参数。

    - **示例**：

      ```java
      public interface UserMapper {
          @Select("SELECT * FROM users WHERE username = #{username} AND status = #{status}")
          List<User> findUsers(@Param("username") String username, @Param("status") String status);
      }
      ```

      在这个例子中，`@Param("username")` 和 `@Param("status")` 分别为方法参数 `username` 和 `status` 命名，这样在SQL语句中可以通过 `#{username}` 和 `#{status}` 来引用这些参数值。

  - `@RequestParam`

    - **所属框架**：Spring MVC

    - **作用位置**：Controller层

    - **用途**：`@RequestParam` 注解用于将请求参数（即URL查询参数或者表单数据）绑定到控制器的方法参数上。它可以用来获取GET请求中的查询参数或POST请求中的表单数据。

    - **示例**：

      ```java
      @RestController
      public class UserController {
      
          @GetMapping("/users")
          public List<User> getUsers(@RequestParam(value = "role", required = false) String role) {
              // 根据role参数查询用户逻辑...
          }
      }
      ```

      这个例子展示了如何从HTTP GET请求中提取名为`role`的查询参数，并将其作为参数传递给`getUsers`方法。

    - 区别与联系

    - **使用场景不同**：`@Param` 主要用于MyBatis的Mapper接口中，帮助动态生成SQL语句；而 `@RequestParam` 则是Spring MVC的一部分，用于处理HTTP请求参数。

    - **作用对象不同**：`@Param` 是对DAO层方法参数进行标注，使其能在SQL映射文件中被引用；`@RequestParam` 是对Controller层方法参数进行标注，用于接收客户端发送过来的请求参数。

    - **技术栈差异**：`@Param` 关联的是数据库访问层面的技术（如MyBatis），而 `@RequestParam` 关联的是Web应用层面的技术（如Spring MVC）。

    尽管两者都涉及到参数的标记和传递，但它们各自服务于不同的目的和技术层次，因此不能直接替换使用。理解它们的作用范围有助于更好地设计和实现分层架构的应用程序。

- 6 prop="username" 为啥可以直接写username，是从哪里可以确定？


  - 这个直接写username，是要与后端进行交互。后端返回的data中必须要有username

- 7 VUE中$refs的基本用法


  - ```
    ref 有三种用法：
    　　1、ref 加在普通的元素上，用this.$refs.（ref值） 获取到的是dom元素
    　　2、ref 加在子组件上，用this.$refs.（ref值） 获取到的是组件实例，可以使用组件的所有方法。在使用方法的时候直接this.$refs.（ref值）.方法（） 就可以使用了。
    　3、如何利用 v-for 和 ref 获取一组数组或者dom 节点
                            
    原文链接：https://blog.csdn.net/wh710107079/article/details/88243638
    ```

    


### 3.课程中遇见的弹幕解决方案，后期验证之后看效果。

- vscode中代码格式化shift+alt+F
- 数字加个空字符串就变成字符了  xx.id+''
- 学习数据库用户加密与解密

# font格式

<font color = 'yellow'>复制修改</font>

<font color = 'pink'>复制修改</font>

<font color = 'orange'>复制修改</font>
