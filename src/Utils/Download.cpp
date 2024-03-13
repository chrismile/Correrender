/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <boost/algorithm/string/replace.hpp>
#include <Utils/Dialog.hpp>

#include "Download.hpp"

// Include Curl after Renderer.hpp, as NaNHandling::IGNORE conflicts with windows.h.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <curl/curl.h>

static size_t writeDataCallbackCurl(void *pointer, size_t size, size_t numMembers, void *stream) {
    size_t written = fwrite(pointer, size, numMembers, (FILE*)stream);
    return written;
}

static bool userGaveDownloadConsent = false;

bool downloadFile(const std::string &url, const std::string &localFileName) {
    if (!userGaveDownloadConsent) {
        auto button = sgl::dialog::openMessageBoxBlocking(
                "Download world map data",
                "Download world map data from naturalearthdata.com?",
                sgl::dialog::Choice::YES_NO, sgl::dialog::Icon::QUESTION);
        if (button != sgl::dialog::Button::YES && button != sgl::dialog::Button::OK) {
            return false;
        }
        userGaveDownloadConsent = true;
    }

    CURL* curlHandle = curl_easy_init();
    if (!curlHandle) {
        return false;
    }
    CURLcode curlErrorCode = CURLE_OK;

    char* compressedUrl = curl_easy_escape(curlHandle, url.c_str(), int(url.size()));
    std::string fixedUrl = compressedUrl;
    boost::replace_all(fixedUrl, "%3A", ":");
    boost::replace_all(fixedUrl, "%2F", "/");
    std::cout << "Starting to download \"" << fixedUrl << "\"..." << std::endl;

    curl_easy_setopt(curlHandle, CURLOPT_URL, fixedUrl.c_str());
    //curl_easy_setopt(curlHandle, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curlHandle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, 1L);
    // TODO: Migrate to naciscdn.org? https://github.com/nvkelso/natural-earth-vector/issues/895
    curl_easy_setopt(curlHandle, CURLOPT_REFERER, "https://www.naturalearthdata.com/downloads/");
    curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, writeDataCallbackCurl);
    FILE* pagefile = fopen(localFileName.c_str(), "wb");
    if (pagefile) {
        curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, pagefile);
        curlErrorCode = curl_easy_perform(curlHandle);
        if (curlErrorCode != CURLE_OK) {
            fclose(pagefile);
            curl_free(compressedUrl);
            curl_easy_cleanup(curlHandle);
            return false;
        }
        fclose(pagefile);
    }

    curl_free(compressedUrl);
    curl_easy_cleanup(curlHandle);
    return true;
}
