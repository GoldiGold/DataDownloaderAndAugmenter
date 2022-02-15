from bs4 import BeautifulSoup
import requests
import os, re

full_path = '/Users/chengoldi/Desktop/university/Masters Degree/Thesis Material/Dataset/100307/T1w/1D Files'


def get_mask(link: str, full_1d_path, file_name):
    """
    we found out that the page is just a "text file" so in order to extract the values we only need to get the
    soup.text and save it to a file in the computer
    :param link:
    :return:
    """
    page = requests.get(link)
    soup = BeautifulSoup(page.content, 'html.parser')
    # print(soup.prettify())
    # mask = soup.find_all("body", class_='vsc-initialized')
    # print(len(mask))  # , mask[0].text[:10])
    if not os.path.isdir(full_1d_path):
        print(f'this {full_1d_path} is NOT a firectory')
    # need to create it
        os.makedirs(full_1d_path)
    mask_file_name = "/".join([full_1d_path, file_name]) + ".txt"
    # print(mask_file_name)
    # return 0
    if not os.path.isfile(mask_file_name):
        # print(f'it is not a file: {mask_file_name}')
        with open(mask_file_name, 'w') as mask:
            # if mask.
            mask.write(soup.text)
    else:
        print(f'file {mask_file_name} already exist')
    return 0


def crawl_links(link: str, patients_parent_path: str, path_1d: str, start='BA', end='1D'):
    page = requests.get(link)
    # print(page.text)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all('a')
    # print(links)
    #
    i = 0
    for l in links:
        next_link = l['href']
        if next_link.endswith(end) and next_link.startswith(start):
            # print(next_link)
            nums = re.findall(r'\d+', next_link)
            # print(nums, [len(idx) for idx in nums])
            brain_id = [idx for idx in nums if len(idx) == 6].pop()  # single element list
            # print(nums)
            mask_link = '/'.join([link, next_link])
            # brain_id = '100307'
            full_1d_path = '/'.join([patients_parent_path, brain_id, path_1d])
            get_mask(mask_link, full_1d_path, next_link)
            # if i == 11:
            #     break
            # i += 1
        # print(next_link)
        # print(link, end="\n")
    # print(soup.prettify())


if __name__ == '__main__':
    brain_nifti_name = 'T1w_acpc_dc_restore_brain.nii.gz'
    brain_id = 100307
    T1w_path_name = f'/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset/{brain_id}/T1w/'
    start_urls = ['http://wwwuser.gwdg.de/~cbsarchi/archiv/public/hcp/']
    crawl_links(start_urls[0], '/run/media/cheng/Maxwell_HD/Goldi_Folder/Dataset',
                'T1w/1D Files')
    print('DONE')
