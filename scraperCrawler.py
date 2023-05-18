# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os
from pathlib import Path
import scrapy
from scrapy.crawler import CrawlerProcess
#
# brain_nifti_name = 'T1w_acpc_dc_restore_brain.nii.gz'
# brain_id = 100307
# T1w_path_name = f'/Users/chengoldi/Desktop/university/Masters Degree/Thesis Material/Dataset/{brain_id}/T1w'


class MaskSpider(scrapy.Spider):
    name = 'mask_spider'
    start_urls = ['http://wwwuser.gwdg.de/~cbsarchi/archiv/public/hcp/']

    def parse_mask(self, response):
        mask_selector = '.vsc-initialized'
        for mask in response.css(mask_selector):
            mask_value = 'pre'

            yield {
                'mask': mask.css(mask_value).extract_first(),

            }

    def parse(self, response):
        """
        first page is the page of all the links of the masks
        second pages (inside the urls) are the masks pages themselves
        so we first go through all the links and then call the second method: parse_mask to read an store it.
        """
        link_selector = 'a ::attr(href)'
        print([link for link in response.css(link_selector)])
        return
        # for link in self.link_extractor.extract_links(response):
        #     yield Request(link.url, callback=self.parse)
        '''
        for link in response.css(link_selector):
            # print(mask)
            NEXT_PAGE_SELECTOR = 'a ::attr(href)'
            next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()
            print(next_page)
            if next_page:
                yield scrapy.Request(
                    response.urljoin(next_page),
                    callback=self.parse_mask
                )
            # pass
        '''

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    # p = Path(T1w_path_name)
    process = CrawlerProcess()
    process.crawl(MaskSpider)
    process.start()
    # ms = MaskSpider()
    # ms.parse()
    # for pp in p.iterdir():
    #     if pp.name.endswith('nii.gz'):
    #         print(pp)
    # print([f'{pp} \n'for pp in p.iterdir()])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
