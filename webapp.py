
sys.path.append('modules')

load_dotenv()

username=os.getenv("USERNAME")
password=os.getenv("PASSWORD")
hostname=os.getenv("HOSTNAME")
port=os.getenv("PORT")
db=os.getenv("DB")
api_key=os.getenv("API_KEY")

cursor,conn=create_conn(hostname,db,username,password,port)

movies_path="resources/movies.csv"
if not os.path.isfile(movies_path):
    df=fetch_current_month_movies_to_df_with_posters(api_key)
    df.to_csv(movies_path, sep='\t')
    print("downloaded tmdb information")
else:
    df = pd.read_csv(movies_path, sep='\t', lineterminator='\n').drop_duplicates(subset=["title"]).dropna().reset_index(drop=True, inplace=False)

movies_path2="resources/movies2.csv"
if not os.path.isfile(movies_path2):
    df2=create_df(cursor).reset_index(drop=True, inplace=False)
    df2.to_csv(movies_path2, sep='\t')
else:
    df2 = pd.read_csv(movies_path2, sep='\t', lineterminator='\n').drop_duplicates().dropna().reset_index(drop=True, inplace=False)


user_list={}


temp_data={None : {"lightmode" : False}}


def get_hash(algorithm, data):
   """Generate a hash for given data using specified algorithm"""
   hash_obj = hashlib.new(algorithm)
   hash_obj.update(data.encode())
   return hash_obj.hexdigest()

def search_string(s, search):
    return search in str(s).lower()

def chckpwd(user,pwd,fakesession):
    global temp_data,cursor,conn,user_list,df,df2
    try : info=super_function(cursor)
    except :
        cursor,conn=create_conn(hostname,db,username,password,port)
        info=super_function(cursor)
    user_list=info[0]
    if user in user_list.keys():
        if user in [temp_data[e]["user"] if "user" in temp_data[e].keys() else None for e in temp_data.keys()]:
            del temp_data[fakesession]
        if get_hash("sha256", pwd)==user_list[user] :
            temp_data[fakesession]={}
            temp_data[fakesession]["user"]=user
            try : temp_data[fakesession]["search"]=info[3][user] #there might be an exception here that is not displayed
            except : temp_data[fakesession]["search"]=[]
            try : 
                temp_data[fakesession]["liked"]=info[1][user]
                temp_data[fakesession]["recommendations"]=get_cosine_sim_recommendations(df, info[1][user], 30, df2)
            except : 
                temp_data[fakesession]["liked"]=[]
                temp_data[fakesession]["recommendations"]=df["title"].head(30).to_list()
            temp_data[fakesession]["lightmode"]=info[2][user]
            return True
    return False

def change_user_lightmode(fakesession):
    global cursor,conn,temp_data
    temp_data[fakesession]["lightmode"]=1-temp_data[fakesession]["lightmode"]
    if "user" in temp_data[fakesession].keys() :
        user=temp_data[fakesession]["user"]
        try :
            change_lightmode(cursor,user,temp_data[fakesession]["lightmode"])
            conn.commit()
        except :
            cursor,conn=create_conn(hostname,db,username,password,port)
            change_lightmode(cursor,user,temp_data[fakesession]["lightmode"])
            conn.commit()


def create_login(user,pwd):
    global user_list,cursor,conn
    try : info=fetch_users(cursor)
    except :
        cursor,conn=create_conn(hostname,db,username,password,port)
        info=fetch_users(cursor)

    d={}
    for e in info:
        d[e[0]]=e[1]
    
    user_list=d
    if not user in user_list.keys():
        hashed_pwd=get_hash("sha256", pwd)
        user_list[user]=hashed_pwd
        create_user(cursor,user,hashed_pwd)
        conn.commit()
        return True
    return False

def update_liked_movies(fakesession,movie):
    global temp_data,conn,cursor,df,df2
    isliked = movie in temp_data[fakesession]["liked"]
    user=temp_data[fakesession]["user"]
    if isliked :
        temp_data[fakesession]["liked"].remove(movie)
    else :
        temp_data[fakesession]["liked"].append(movie)
    try :
        write_favourite(cursor,user,movie,isliked)
        conn.commit()
    except :
        cursor,conn=create_conn(hostname,db,username,password,port)
        write_favourite(cursor,user,movie,isliked)
        conn.commit()
    if len(temp_data[fakesession]["liked"])!=0:
        temp_data[fakesession]["recommendations"]=get_cosine_sim_recommendations(df, temp_data[fakesession]["liked"], 30, df2)
    else: 
        temp_data[fakesession]["recommendations"]=df["title"].head(30).to_list()


def update_history(fakesession,movie):
    global temp_data,conn,cursor
    user=temp_data[fakesession]["user"]
    try :
        write_history(cursor,user,movie)
        conn.commit()
    except :
        cursor,conn=create_conn(hostname,db,username,password,port)
        write_history(cursor,user,movie)
        conn.commit()


def reset_stgs(lightmode):
    """used to specify the default settings used for the first connection"""
    global df
    if lightmode : colors = [(230, 230, 230),(180, 180, 180),(169, 77, 255),(206, 153, 255),(31, 34, 45),(20,20,20),"light"] #for lightmode
    else : colors = [(44, 46, 63),(74, 77, 105),(99, 0, 192),(124, 0, 240),(31, 34, 45),(255,255,255),"dark"]
    file=open('resources/words.txt', 'r')
    autocomplete=df["title"].tolist()
    return colors,autocomplete





#FastAPI and sass parameters
app = FastAPI()
sass.compile(dirname=('asset', 'static'))
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
from custom_url_processor import *
templates.env.globals['CustomURLProcessor'] = CustomURLProcessor






def generic_render(request,fakesession,dictionary,name,title):
    """handles the light/dark modes for you, uses a dictionary to pass information to the pages"""
    global temp_data
    if "user" in temp_data[fakesession].keys():
        logged="Logged in"
    else : logged="Logged off"
    colors=reset_stgs(temp_data[fakesession]["lightmode"])[0]
    return templates.TemplateResponse(
        request=request, name=name, context={"fakesession" : fakesession ,"state" : logged, "dictionary": dictionary, "colors" : colors, "title" : title}
    )




#special routes

@app.get("/create_cookie") #creates a cookie and redirects to /home
async def create_cookie(response: Response):
    global temp_data
    colors,autocomplete=reset_stgs(False)
    uuid=str(uuid4())
    temp_data[uuid]={}
    temp_data[uuid]["lightmode"]=False
    temp_data[uuid]["search"]=[]
    temp_data[uuid]["liked"]=[]
    temp_data[uuid]["recommendations"]=df["title"].head(30).to_list()
    response=RedirectResponse(url="/home")
    response.set_cookie(key="fakesession", value=uuid, httponly=True)
    return response

@app.get("/logout")
async def logout(response: Response, fakesession: Union[str, None] = Cookie(default=None)):
    response=RedirectResponse(url="/create_cookie")
    del temp_data[fakesession]
    response.delete_cookie(key="fakesession", httponly=True)
    return response

@app.get("/del_cookie")
async def del_cookie(response: Response, fakesession: Union[str, None] = Cookie(default=None)):
    response=RedirectResponse(url="/home")
    del temp_data[fakesession]
    response.delete_cookie(key="fakesession", httponly=True)
    return response

@app.get("/") #redirects to /create_cookie
async def index(response: Response):
    return RedirectResponse(url="/create_cookie")










#routes for the all the pages

@app.get("/home", response_class=HTMLResponse)
async def home(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None)):
    global temp_data, df
    autocomplete=reset_stgs(False)[1]
    movies={"titles":[e.replace(" ", "__").replace("/","_slash_") for e in df["title"].head(30).tolist()],"poster":df["poster_url"].head(30).tolist()}
    if fakesession!=None :
        if "user" in temp_data[fakesession].keys() :
            recommendations=temp_data[fakesession]["recommendations"]
            if len(recommendations)!=0:
                titles=[e.replace(" ", "__").replace("/","_slash_") for e in recommendations]
                movies={"titles":titles,"poster":[df[df["title"]==e]["poster_url"].tolist()[0] for e in recommendations]}
    return generic_render(request,fakesession,{"autocomplete":autocomplete,"movies":movies,"amout":30,"h_text":"Just for you :"},"index.html","Webfloox")

@app.post("/home", response_class=HTMLResponse)
async def home(request: Request, response: Response, switchmode: Annotated[str, Form()], fakesession: Union[str, None] = Cookie(default=None)):
    global temp_data, df
    if switchmode == "1" and fakesession!=None : change_user_lightmode(fakesession)
    autocomplete=reset_stgs(False)[1]
    movies={"titles":[e.replace(" ", "__").replace("/","_slash_") for e in df["title"].head(30).tolist()],"poster":df["poster_url"].head(30).tolist()}
    if fakesession!=None :
        if "user" in temp_data[fakesession].keys() :
            recommendations=temp_data[fakesession]["recommendations"]
            if len(recommendations)!=0:
                titles=[e.replace(" ", "__").replace("/","_slash_") for e in recommendations]
                movies={"titles":titles,"poster":[df[df["title"]==e]["poster_url"].tolist()[0] for e in recommendations]}
    return generic_render(request,fakesession,{"autocomplete":autocomplete,"movies":movies,"amout":30,"h_text":"Just for you :"},"index.html","Webfloox")





@app.get("/favourites", response_class=HTMLResponse)
async def favourites(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None)):
    global temp_data, df
    autocomplete=reset_stgs(False)[1]
    if fakesession!=None :
        if "user" in temp_data[fakesession].keys() :
            liked=temp_data[fakesession]["liked"]
            titles=[e.replace(" ", "__").replace("/","_slash_") for e in liked]
            movies={"titles":titles,"poster":[df[df["title"]==e]["poster_url"].tolist()[0] for e in liked]}
            return generic_render(request,fakesession,{"autocomplete":autocomplete,"movies":movies,"amout":len(liked),"h_text":"Your favourites :"},"index.html","Webfloox - favourites")
    return RedirectResponse(url="/home")

@app.post("/favourites", response_class=HTMLResponse)
async def favourites(request: Request, response: Response, switchmode: Annotated[str, Form()], fakesession: Union[str, None] = Cookie(default=None)):
    global temp_data, df
    if switchmode == "1" and fakesession!=None : change_user_lightmode(fakesession)
    autocomplete=reset_stgs(False)[1]
    if fakesession!=None :
        if "user" in temp_data[fakesession].keys() :
            liked=temp_data[fakesession]["liked"]
            titles=[e.replace(" ", "__").replace("/","_slash_") for e in liked]
            movies={"titles":titles,"poster":[df[df["title"]==e]["poster_url"].tolist()[0] for e in liked]}
            return generic_render(request,fakesession,{"autocomplete":autocomplete,"movies":movies,"amout":len(liked),"h_text":"Your favourites :"},"index.html","Webfloox - favourites")
    return RedirectResponse(url="/home")












@app.post("/results", response_class=HTMLResponse)
async def results(request: Request, search: Annotated[str, Form()] = "", switchmode: Annotated[str, Form()] = None, fakesession: Union[str, None] = Cookie(default=None)):
    """This is a test function for the navbar search module, it changes the pages title acording to the search input"""
    global temp_data
    if fakesession!=None :
        if search == "" : search=temp_data[fakesession]["search"][-1]
        else : temp_data[fakesession]["search"].append(search)
        if "user" in temp_data[fakesession].keys() : update_history(fakesession,search)
    if switchmode == "1" and fakesession!=None : change_user_lightmode(fakesession)
    mask = df[["title","overview"]].apply(lambda x: x.map(lambda s: search_string(s, search)))
    df0=df.loc[mask.any(axis=1)]
    movies={"titles":[e.replace(" ", "__").replace("/","_slash_") for e in df0["title"].tolist()],"poster":df0["poster_url"].tolist(),"over":df0["overview"].tolist(),"date":df0["release_date"].tolist()}
    autocomplete=reset_stgs(False)[1]
    return generic_render(request,fakesession,{"autocomplete":autocomplete,"search":search,"movies":movies,"length":min(len(movies["titles"]),200)},"results.html","Results for " + search)

@app.get("/results")
async def results(response: Response):
    return RedirectResponse(url="/home")







@app.get("/whoami") #simply a test route to check the fakesession cookie
async def whoami(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None), user_agent: Annotated[str | None, Header()] = None):
    global temp_data
    client=str(request.client)
    autocomplete=reset_stgs(False)[1]
    user = None
    liked = []
    if fakesession!=None :
        search=temp_data[fakesession]["search"]
        if "user" in temp_data[fakesession].keys() :
            user=temp_data[fakesession]["user"]
            liked=temp_data[fakesession]["liked"]
        return generic_render(request,fakesession,{"autocomplete":autocomplete,"user_agent":user_agent,"client":client, "search":search,"user":user,"liked":liked},"whoami.html","Webfloox - Who am I ?")
    else : return generic_render(request,fakesession,{"autocomplete":autocomplete,"user_agent":user_agent,"client":client,"user":user,"liked":liked},"whoami.html","Webfloox - Who am I ?")


@app.post("/whoami") #simply a test route to check the fakesession cookie
async def whoami(request: Request, response: Response, switchmode: Annotated[str, Form()], fakesession: Union[str, None] = Cookie(default=None), user_agent: Annotated[str | None, Header()] = None):
    global temp_data
    client=str(request.client)
    if switchmode == "1" and fakesession!=None : change_user_lightmode(fakesession)
    autocomplete=reset_stgs(False)[1]
    user = None
    liked = []
    if fakesession!=None :
        search=temp_data[fakesession]["search"]
        if "user" in temp_data[fakesession].keys() :
            user=temp_data[fakesession]["user"]
            liked=temp_data[fakesession]["liked"]
        return generic_render(request,fakesession,{"autocomplete":autocomplete,"user_agent":user_agent,"client":client, "search":search,"user":user,"liked":liked},"whoami.html","Webfloox - Who am I ?")
    else : return generic_render(request,fakesession,{"autocomplete":autocomplete,"user_agent":user_agent,"client":client,"user":user,"liked":liked},"whoami.html","Webfloox - Who am I ?")


@app.get("/login", response_class=HTMLResponse)
async def login(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None)):
    return generic_render(request,fakesession,{"wrong":False},"login.html","Webfloox - Log in")

@app.post("/login", response_class=HTMLResponse)
async def login(request: Request, response: Response, user: Annotated[str, Form()], pwd: Annotated[str, Form()], fakesession: Union[str, None] = Cookie(default=None)):
    global temp_data
    testlogin=chckpwd(user,pwd,fakesession)
    if testlogin :
        temp_data[fakesession]["user"]=user
        return RedirectResponse(url="/home", status_code=status.HTTP_303_SEE_OTHER)
    else : return generic_render(request,fakesession,{"wrong":True},"login.html","Webfloox - Log in")



@app.get("/signup", response_class=HTMLResponse)
async def signup(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None)):
    return generic_render(request,fakesession,{"wrong":False},"signup.html","Webfloox - Sign up")

@app.post("/signup", response_class=HTMLResponse)
async def signup(request: Request, response: Response, user: Annotated[str, Form()],pwd: Annotated[str, Form()], pwd2: Annotated[str, Form()], fakesession: Union[str, None] = Cookie(default=None)):
    global temp_data
    test=create_login(user,pwd)
    if test :
        temp_data[fakesession]["user"]=user
        return RedirectResponse(url="/home", status_code=status.HTTP_303_SEE_OTHER)
    else : return generic_render(request,fakesession,{"wrong":True},"signup.html","Webfloox - Sign up")







@app.get("/movie/{movie}", response_class=HTMLResponse)
async def movie(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None),movie:str="test"):
    global temp_data,df
    movie_og=''.join([s for s in movie])
    movie=movie.replace("__", " ").replace("_slash_","/")
    poster=df[df["title"]==movie]["poster_url"].tolist()[0]
    over=df[df["title"]==movie]["overview"].tolist()[0]
    date=df[df["title"]==movie]["release_date"].tolist()[0]
    liked=False
    if fakesession != None:
        if "user" in temp_data[fakesession]:liked = movie in temp_data[fakesession]["liked"]
    autocomplete=reset_stgs(False)[1]
    return generic_render(request,fakesession,{"autocomplete":autocomplete,"movie":movie,"poster":poster,"over":over,"liked":liked,"movie_og":movie_og,"date":date},"movie.html",f"Webfloox - {movie}")

@app.post("/movie/{movie}", response_class=HTMLResponse)
async def movie(request: Request, response: Response, switchmode: Annotated[str, Form()], fakesession: Union[str, None] = Cookie(default=None),movie:str="test"):
    global temp_data,df
    movie_og=''.join([s for s in movie])
    movie=movie.replace("__", " ").replace("_slash_","/")
    poster=df[df["title"]==movie]["poster_url"].tolist()[0]
    over=df[df["title"]==movie]["overview"].tolist()[0]
    date=df[df["title"]==movie]["release_date"].tolist()[0]
    if switchmode == "1" and fakesession!=None : temp_data[fakesession]["lightmode"]=1-temp_data[fakesession]["lightmode"]
    liked=False
    if fakesession != None:
        if "user" in temp_data[fakesession]:liked = movie in temp_data[fakesession]["liked"]
    autocomplete=reset_stgs(False)[1]
    return generic_render(request,fakesession,{"autocomplete":autocomplete,"movie":movie,"poster":poster,"over":over,"liked":liked,"movie_og":movie_og,"date":date},"movie.html",f"Webfloox - {movie}")

@app.get("/like_function/{movie}", response_class=HTMLResponse)
async def like_function(request: Request, response: Response, fakesession: Union[str, None] = Cookie(default=None),movie:str="test"):
    global temp_data,df
    movie_copy=''.join([s for s in movie]).replace("__", " ").replace("_slash_","/")
    liked=False
    if fakesession != None:
        if "user" in temp_data[fakesession]:
            update_liked_movies(fakesession,movie_copy)

    return RedirectResponse(url=f"/movie/{movie}", status_code=status.HTTP_303_SEE_OTHER)



"""
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)"""
