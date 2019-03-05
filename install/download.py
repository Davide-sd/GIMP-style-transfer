import requests
import os
import shutil
import tarfile

thisFolder = os.path.dirname(os.path.abspath(__file__))
pluginFolderPath = os.path.dirname(thisFolder)

# https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive#39225039
def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def download_file(url, destination, text=False):
    filename = destination.split('/')[-1]
    with open(destination, 'wb') as f:
        if text:
            with requests.get(url) as r:
                f.write(r.text)
        else:
            with requests.get(url, stream=True) as r:
                shutil.copyfileobj(r.raw, f)
    print("\t{0} downloaded!".format(filename))

def print_header(n):
    msg = " IMPLEMENTATION {0} SETUP ".format(n)
    n1 = 70
    n2 = int((n1 - len(msg)) / 2)
    print("".join(["#" for i in range(n1)]))
    print("".join(["#" for i in range(n2)]) + msg + "".join(["#" for j in range(n2)]))
    print("".join(["#" for i in range(n1)]))

def implementation_1():
    print_header(1)

    # download and save transform.py to the folder implementation_1/src
    url = "https://raw.githubusercontent.com/lengstrom/fast-style-transfer/master/src/transform.py"
    destination = os.path.join(pluginFolderPath, "implementation_1/src/transform.py")
    if not os.path.exists(destination):
        print("\nDownloading file {0} into folder {1}".format(url, destination))
        download_file(url, destination, True)

    # In this dictionary:
    # the key is displayed in the print statement
    # value[0]: filename of the model to be saved
    # value[1]: link ID for google drive
    fast_style_transfer = {
        "The Scream": ["the-scream", "0B9jhaT37ydSyZ0RyTGU0Q2xiU28"],
        "La Muse": ["la-muse", "0B9jhaT37ydSyQU1sYW02Sm9kV3c"],
        "Udnie": ["udnie", "0B9jhaT37ydSyb0NuYmk2ZEpOR0E"],
        "Great Wave": ["great-wave", "0B9jhaT37ydSyVGk0TC10bDF1S28"],
        "Rain Princess": ["rain-princess", "0B9jhaT37ydSyaEJlSFlIeUxweGs"],
        "The Shipwreck of the minotaur": ["the-shipwreck", "0B9jhaT37ydSySjNrM3J5N2gweVk"]
    }

    print("""\nWARNING: attempting to download the model files (115MB of free
    disk space needed.)""")
    i = 0
    for k, v in fast_style_transfer.items():
        print("{0}/{1}\tDownloading the model for '{2}'".format(i+1, len(fast_style_transfer.keys()), k))
        filename = v[0] + ".ckpt"
        destination_path = os.path.join(pluginFolderPath, "implementation_1/models", filename)
        if not os.path.exists(destination_path):
            download_file_from_google_drive(v[1], destination_path)
            print("\t{0} downloaded!".format(filename))
        else:
            print("\tThe model {0} already exists. Skipping this download.".format(filename))
        i += 1

    print("\nImplementation_1 (Style Transfer) is ready to be used!!!")

def implementation_2():
    # url for the online folder containing the models (+ download path)
    base_url = "https://hcicloud.iwr.uni-heidelberg.de/index.php/s/XXVKT5grAquXNqi/download?path=%2F&files="

    # In this dictionary:
    # the key is displayed in the print statement
    # value: filename of the model to be saved
    artist_style_transfer = {
        "Berthe Morisot": "model_morisot_ckpt.tar.gz",
        "Claude Monet": "model_monet_ckpt.tar.gz",
        "Edvard Munch": "model_munch_ckpt.tar.gz",
        "El Greco": "model_el-greco_ckpt.tar.gz",
        "Ernst Ludwig Kirchner": "model_kirchner_ckpt.tar.gz",
        "Jackson Pollock": "model_pollock_ckpt.tar.gz",
        "Nicholas Roerich": "model_roerich_ckpt.tar.gz",
        "Pablo Picasso": "model_picasso_ckpt.tar.gz",
        "Paul Cezanne": "model_cezanne_ckpt.tar.gz",
        "Paul Gauguin": "model_gauguin_ckpt.tar.gz",
        "Samuel Peploe": "model_peploe_ckpt.tar.gz",
        "Vincent Van Gogh": "model_van-gogh_ckpt.tar.gz",
        "Wassily Kandisky": "model_kandinsky_ckpt.tar.gz"
    }

    artists = artist_style_transfer.keys()
    artists.sort()

    # create temporary download folder
    tmp_folder = os.path.join(thisFolder, "tmp")
    if not os.path. exists(tmp_folder):
        os.makedirs(tmp_folder)

    print_header(2)

    # chose which model to download
    print("""\n WARNING: the download of all the artists is going to use 9.1GB
    of disk space and may take a while. Here you have the possibility to select
    only the artists you are interested in.""")

    print("\t-1\tAll artists (default)")
    for i, a in enumerate(artists):
        print("\t{0}\t{1}".format(i, a))

    vals = raw_input("""Insert the numbers associated to the interested artists.
    Use comma to separate them. Default value -1 (all artists): """)

    if vals == "":
        vals = "-1"

    # keep only integers
    parsed_values = []
    for v in vals.replace(" ", "").split(","):
        try:
            parsed_values.append(int(v))
        except ValueError:
            print("Unable to convert '{0}' into a 'int'. Skipping it!".format(v))

    parsed_values = set(parsed_values)
    if len(parsed_values) > 1 and -1 in parsed_values:
        parsed_values.discard(-1)
        print("""You decided to download only specific artists. Skipping -1 (All
        Artists) option!""")
    parsed_values = list(parsed_values)

    artists_to_process = []
    if len(parsed_values) == 1 and parsed_values[0] == -1:
        artists_to_process = artists
    else:
        artists_to_process = [artists[v] for v in parsed_values]

    # move to implementation_2/models in order to extract the archives
    os.chdir(os.path.join(pluginFolderPath, "implementation_2/models"))

    for i, k in enumerate(artists_to_process):
        v = artist_style_transfer[k]
        model = v.replace("_ckpt", "").split(".")[0]
        model_folder = os.path.join(pluginFolderPath, "implementation_2/models", model)
        if not os.path.exists(model_folder):
            # download
            print("{0}/{1}\tDownloading the model archive for '{2}'".format(i+1, len(artists_to_process), k))
            destination_path = os.path.join(tmp_folder, v)
            if not os.path.exists(destination_path):
                download_file(base_url + v, destination_path)
                pass
            else:
                print("\tThe model {0} already exists. Skipping this download.".format(filename))

            # extract
            print("\tExtracting {0}...".format(v))
            tar = tarfile.open(destination_path, "r:gz")
            tar.extractall()
            tar.close()
            print("\tExtraction complete! Removing temporary file {0}.".format(v))
            # delete
            os.remove(destination_path)
        else:
            print("{0}/{1}\t{2} already exists. Skipping download!!!".format(i+1, len(artists_to_process), model))

    print("\nImplementation_2 (Artist Style Transfer) is ready to be used!!!")

def implementation_3():
    print_header(3)

    print("""\nWARNING: attempting to download the model files (40.2MB of free
    disk space needed.)\n""")

    urls = [
        "https://github.com/tensorlayer/pretrained-models/raw/master/models/style_transfer_models_and_examples/pretrained_vgg19_decoder_model.npz",
        "https://github.com/tensorlayer/pretrained-models/raw/master/models/style_transfer_models_and_examples/pretrained_vgg19_encoder_model.npz"
    ]

    folder = os.path.join(pluginFolderPath, "implementation_3/models")
    for i, url in enumerate(urls):
        name = url.split("/")[-1]
        filename = os.path.join(folder, name)
        if not os.path.exists(filename):
            print("{0}/{1}\t Downloading {2}".format(i+1, len(urls), name))
            download_file(url, filename)
        else:
            print("{0} already exists. Skipping download.".format(name))

    print("\nImplementation_3 (Arbitrary Style Transfer) is ready to be used!!!")

if __name__ == "__main__":
    def print_error_msg(n, url):
        print("""\nSomething went wrong with the setup of Implementation_{0} (Style
        Transfer). Please, try to follow the manual procedure:
        {1}""".format(n, url))

    try:
        implementation_1()
    except:
        print_error_msg(1, "https://github.com/Davide-sd/GIMP-style-transfer#setting-up-style-transfer")

    try:
        implementation_2()
    except:
        print_error_msg(2, "https://github.com/Davide-sd/GIMP-style-transfer#setting-up-artist-style-transfer")

    try:
        implementation_3()
    except:
        print_error_msg(3, "https://github.com/Davide-sd/GIMP-style-transfer#setting-up-arbitrary-style-transfer")
