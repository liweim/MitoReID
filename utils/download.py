import hashlib
import os
import os.path as osp
import shutil
import sys
import tempfile
import textwrap
import time
import urllib.parse
from http.cookiejar import MozillaCookieJar
import bs4
import requests
import tqdm
import re
import urllib.parse
import warnings
import argparse

CHUNK_SIZE = 512 * 1024  # 512KB
home = osp.expanduser("~")
name2id = {
    'Mito.zip': '1_GHrxZFUsG7NeSLDRn9M3nv2iCSY5rXk',
    'resnet-18-kinetics.pth': '1jOsaPJw9Qcu5uEqmkZYhdozsOVTJkUjs',
    'ide.pth': '1dHJ92huqqiW2YCT51JMGRzN43QSq9KtD',
    'resnet-18-mito.pth': '1QgBON_t8WnThfrio5uvp4InJHkZoeSTL',
    'mitoreid-resnet-18.pth': '1YcNvnkgIGbB2a-_P-EhTTFKFhl4E3N1W',
    'mitoreid-resnet-18-standard.pth': '1y55yaTvkR79fbvPayAAdjjFaxVYDecO9',
    'mitoreid-resnet-18-image.pth': '1a3O1-3NVeleK_uvHzWh9g_IQVMOBJgkf',
    'mitoreid-resnet-18-pretrain.pth': '1jZpWot9x7BxNL8__snRj_LYwTyWK4L3M',
    'resnet-50-kinetics.pth': '1oaZ1gjYpcij2ueRhROXsq8MA00R7AQhB',
    'resnet-50-mito.pth': '1pmpInelhULHaLrs8ygY478XWHbYrOtLU',
    'mitoreid-resnet-50.pth': '1QpOsqGwSpJQsmoNeuod2pI5_wfS-zkVq',
    'mitoreid-resnet-50-standard.pth': '1MCv7MQk68Jw_uO4IVeYgbs54pgRJQzja',
    'mitoreid-resnet-50-image.pth': '1cwyL908uwq1vgP9T7N32xCtuqoPdtAye',
    'mitoreid-resnet-50-pretrain.pth': '1oweWEp1LwklOJwwPdNcRMZTuS9h0DEHk',
}


def indent(text, prefix):
    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if line.strip() else line)

    return "".join(prefixed_lines())


def is_google_drive_url(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname in ["drive.google.com", "docs.google.com"]


def parse_url(url, warning=True):
    """Parse URLs especially for Google Drive links.

    file_id: ID of file on Google Drive.
    is_download_link: Flag if it is download link of Google Drive.
    """
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    is_gdrive = is_google_drive_url(url=url)
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
        return is_gdrive, is_download_link

    file_id = None
    if "id" in query:
        file_ids = query["id"]
        if len(file_ids) == 1:
            file_id = file_ids[0]
    else:
        patterns = [
            r"^/file/d/(.*?)/(edit|view)$",
            r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$",
            r"^/document/d/(.*?)/(edit|htmlview|view)$",
            r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/d/(.*?)/(edit|htmlview|view)$",
            r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$",
            r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, parsed.path)
            if match:
                file_id = match.groups()[0]
                break

    if warning and not is_download_link:
        warnings.warn(
            "You specified a Google Drive link that is not the correct link "
            "to download a file. You might want to try `--fuzzy` option "
            "or the following url: {url}".format(
                url="https://drive.google.com/uc?id={}".format(file_id)
            )
        )

    return file_id, is_download_link


class FileURLRetrievalError(Exception):
    pass


class FolderContentsMaximumLimitError(Exception):
    pass


def get_url_from_gdrive_confirmation(contents):
    url = ""
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = "https://docs.google.com" + m.groups()[0]
            url = url.replace("&amp;", "&")
            break
        soup = bs4.BeautifulSoup(line, features="html.parser")
        form = soup.select_one("#download-form")
        if form is not None:
            url = form["action"].replace("&amp;", "&")
            url_components = urllib.parse.urlsplit(url)
            query_params = urllib.parse.parse_qs(url_components.query)
            for param in form.findChildren("input", attrs={"type": "hidden"}):
                query_params[param["name"]] = param["value"]
            query = urllib.parse.urlencode(query_params, doseq=True)
            url = urllib.parse.urlunsplit(url_components._replace(query=query))
            break
        m = re.search('"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace("\\u003d", "=")
            url = url.replace("\\u0026", "&")
            break
        m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
        if m:
            error = m.groups()[0]
            raise FileURLRetrievalError(error)
    if not url:
        raise FileURLRetrievalError(
            "Cannot retrieve the public link of the file. "
            "You may need to change the permission to "
            "'Anyone with the link', or have had many accesses. "
            "Check FAQ in https://github.com/wkentaro/gdown?tab=readme-ov-file#faq.",
        )
    return url


def _get_filename_from_response(response):
    content_disposition = urllib.parse.unquote(response.headers["Content-Disposition"])

    m = re.search(r"filename\*=UTF-8''(.*)", content_disposition)
    if m:
        filename = m.groups()[0]
        return filename.replace(osp.sep, "_")

    m = re.search('attachment; filename="(.*?)"', content_disposition)
    if m:
        filename = m.groups()[0]
        return filename

    return None


def _get_session(proxy, use_cookies, user_agent, return_cookies_file=False):
    sess = requests.session()

    sess.headers.update({"User-Agent": user_agent})

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        print("Using proxy:", proxy, file=sys.stderr)

    # Load cookies if exists
    cookies_file = osp.join(home, ".cache/gdown/cookies.txt")
    if use_cookies and osp.exists(cookies_file):
        cookie_jar = MozillaCookieJar(cookies_file)
        cookie_jar.load()
        sess.cookies.update(cookie_jar)

    if return_cookies_file:
        return sess, cookies_file
    else:
        return sess


def download(
        url=None,
        output=None,
        quiet=False,
        proxy=None,
        speed=None,
        use_cookies=False,
        verify=True,
        id=None,
        fuzzy=False,
        resume=False,
        format=None,
        user_agent=None,
        log_messages=None,
):
    """Download file from URL.

    Parameters
    ----------
    url: str
        URL. Google Drive URL is also supported.
    output: str
        Output filename. Default is basename of URL.
    quiet: bool
        Suppress terminal output. Default is False.
    proxy: str
        Proxy.
    speed: float
        Download byte size per second (e.g., 256KB/s = 256 * 1024).
    use_cookies: bool
        Flag to use cookies. Default is True.
    verify: bool or string
        Either a bool, in which case it controls whether the server's TLS
        certificate is verified, or a string, in which case it must be a path
        to a CA bundle to use. Default is True.
    id: str
        Google Drive's file ID.
    fuzzy: bool
        Fuzzy extraction of Google Drive's file Id. Default is False.
    resume: bool
        Resume the download from existing tmp file if possible.
        Default is False.
    format: str, optional
        Format of Google Docs, Spreadsheets and Slides. Default is:
            - Google Docs: 'docx'
            - Google Spreadsheet: 'xlsx'
            - Google Slides: 'pptx'
    user_agent: str, optional
        User-agent to use in the HTTP request.
    log_messages: dict, optional
        Log messages to customize. Currently it supports:
        - 'start': the message to show the start of the download
        - 'output': the message to show the output filename

    Returns
    -------
    output: str
        Output filename.
    """
    if not (id is None) ^ (url is None):
        raise ValueError("Either url or id has to be specified")
    if id is not None:
        url = "https://drive.google.com/uc?id={id}".format(id=id)
    if user_agent is None:
        # We need to use different user agent for file download c.f., folder
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"  # NOQA: E501
    if log_messages is None:
        log_messages = {}

    url_origin = url

    sess, cookies_file = _get_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        return_cookies_file=True,
    )

    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        # overwrite the url with fuzzy match of a file id
        url = "https://drive.google.com/uc?id={id}".format(id=gdrive_file_id)
        url_origin = url
        is_gdrive_download_link = True

    while True:
        res = sess.get(url, stream=True, verify=verify)

        if not (gdrive_file_id and is_gdrive_download_link):
            break

        if url == url_origin and res.status_code == 500:
            # The file could be Google Docs or Spreadsheets.
            url = "https://drive.google.com/open?id={id}".format(id=gdrive_file_id)
            continue

        if res.headers["Content-Type"].startswith("text/html"):
            m = re.search("<title>(.+)</title>", res.text)
            if m and m.groups()[0].endswith(" - Google Docs"):
                url = (
                    "https://docs.google.com/document/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="docx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Sheets"):
                url = (
                    "https://docs.google.com/spreadsheets/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="xlsx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Slides"):
                url = (
                    "https://docs.google.com/presentation/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="pptx" if format is None else format,
                    )
                )
                continue
        elif (
                "Content-Disposition" in res.headers
                and res.headers["Content-Disposition"].endswith("pptx")
                and format not in {None, "pptx"}
        ):
            url = (
                "https://docs.google.com/presentation/d/{id}/export"
                "?format={format}".format(
                    id=gdrive_file_id,
                    format="pptx" if format is None else format,
                )
            )
            continue

        if use_cookies:
            cookie_jar = MozillaCookieJar(cookies_file)
            for cookie in sess.cookies:
                cookie_jar.set_cookie(cookie)
            cookie_jar.save()

        if "Content-Disposition" in res.headers:
            # This is the file
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except FileURLRetrievalError as e:
            message = (
                "Failed to retrieve file url:\n\n{}\n\n"
                "You may still be able to access the file from the browser:"
                "\n\n\t{}\n\n"
                "but Gdown can't. Please check connections and permissions."
            ).format(
                indent("\n".join(textwrap.wrap(str(e))), prefix="\t"),
                url_origin,
            )
            raise FileURLRetrievalError(message)

    filename_from_url = None
    if gdrive_file_id and is_gdrive_download_link:
        filename_from_url = _get_filename_from_response(response=res)
    if filename_from_url is None:
        filename_from_url = osp.basename(url)

    if output is None:
        output = filename_from_url

    output_is_path = isinstance(output, str)
    if output_is_path and output.endswith(osp.sep):
        if not osp.exists(output):
            os.makedirs(output)
        output = osp.join(output, filename_from_url)

    if output_is_path:
        existing_tmp_files = []
        for file in os.listdir(osp.dirname(output) or "."):
            if file.startswith(osp.basename(output)):
                existing_tmp_files.append(osp.join(osp.dirname(output), file))
        if resume and existing_tmp_files:
            if len(existing_tmp_files) != 1:
                print(
                    "There are multiple temporary files to resume:",
                    file=sys.stderr,
                )
                print("\n")
                for file in existing_tmp_files:
                    print("\t", file, file=sys.stderr)
                print("\n")
                print(
                    "Please remove them except one to resume downloading.",
                    file=sys.stderr,
                )
                return
            tmp_file = existing_tmp_files[0]
        else:
            resume = False
            # mkstemp is preferred, but does not work on Windows
            # https://github.com/wkentaro/gdown/issues/153
            tmp_file = tempfile.mktemp(
                suffix=tempfile.template,
                prefix=osp.basename(output),
                dir=osp.dirname(output),
            )
        f = open(tmp_file, "ab")
    else:
        tmp_file = None
        f = output

    if tmp_file is not None and f.tell() != 0:
        headers = {"Range": "bytes={}-".format(f.tell())}
        res = sess.get(url, headers=headers, stream=True, verify=verify)

    if not quiet:
        print(log_messages.get("start", f"Downloading {osp.split(output)[1]} ...\n"), file=sys.stderr, end="")
        if resume:
            print("Resume:", tmp_file, file=sys.stderr)

    try:
        total = res.headers.get("Content-Length")
        if total is not None:
            total = int(total)
        if not quiet:
            pbar = tqdm.tqdm(total=total, unit="B", unit_scale=True)
        t_start = time.time()
        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            if not quiet:
                pbar.update(len(chunk))
            if speed is not None:
                elapsed_time_expected = 1.0 * pbar.n / speed
                elapsed_time = time.time() - t_start
                if elapsed_time < elapsed_time_expected:
                    time.sleep(elapsed_time_expected - elapsed_time)
        if not quiet:
            pbar.close()
        if tmp_file:
            f.close()
            shutil.move(tmp_file, output)
    finally:
        sess.close()

    return output


def generate_file_hash(file_path, show_hash=True):
    with open(file_path, 'rb') as file:
        file_content = file.read()

    file_hash = hashlib.sha256(file_content).hexdigest()
    if show_hash:
        print(file_hash)
    return file_hash


def download_file(name, save_folder):
    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    file_path = osp.join(save_folder, name)
    if not osp.exists(file_path):
        download(id=name2id[name], output=file_path, quiet=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', default='Mito.zip', type=str)
    parser.add_argument('--save_folder', default='./result', type=str)
    args = parser.parse_args()

    download_file(args.file_name, args.save_folder)
